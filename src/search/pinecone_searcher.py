"""Pinecone 검색기"""
import os
from typing import Dict, Any, List, Optional
from pinecone import Pinecone
import logging

logger = logging.getLogger(__name__)


class PineconePanelSearcher:
    """Pinecone 벡터DB 검색 (전체 topic 메타데이터 필터 지원 + Fallback)"""

    def __init__(self, pinecone_api_key: str, index_name: str, category_config: Dict[str, Any]):
        """
        Args:
            pinecone_api_key: Pinecone API 키
            index_name: Pinecone 인덱스 이름
            category_config: 카테고리 설정 딕셔너리
        """
        self.category_config = category_config

        # Pinecone 초기화
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)

        logger.info(f"Pinecone 검색기 초기화 완료: {index_name}")

    def get_available_panels(self) -> List[str]:
        """
        사용 가능한 패널 목록 조회
        
        Note: Pinecone은 단일 인덱스 구조이므로 모든 패널을 한 번에 검색 가능
        이 메서드는 호환성을 위해 유지하지만 실제로는 사용되지 않음
        """
        # Pinecone에서는 패널 목록을 미리 조회할 필요가 없음
        # 검색 시 필터로 처리
        return []

    def _is_no_response(self, text: str) -> bool:
        """텍스트가 무응답인지 확인"""
        no_response_patterns = [
            "무응답", "응답하지 않았", "정보 없음", "해당 없음",
            "해당사항 없음", "기록 없음", "데이터 없음"
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in no_response_patterns)

    def _build_filter_condition(self, key: str, value: Any) -> Any:
        """
        Pinecone 필터 조건 생성 (리스트는 $in 연산자로 변환)

        Args:
            key: 메타데이터 키
            value: 단일 값 또는 리스트

        Returns:
            Pinecone 필터 조건
            - 단일 값: value
            - 리스트: {"$in": value}
        """
        if isinstance(value, list) and len(value) > 0:
            # 리스트인 경우 $in 연산자 사용
            return {"$in": value}
        else:
            # 단일 값인 경우 그대로 사용
            return value

    def search_by_category(
        self,
        query_embedding: List[float],
        category: str,
        top_k: int,
        filter_mb_sns: List[str] = None,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        특정 카테고리로 Pinecone 검색 (메타데이터 필터 + Fallback 지원)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            category: 검색할 카테고리 (예: "기본정보", "직업소득", "자동차")
            top_k: 검색 결과 개수
            filter_mb_sns: 필터링할 mb_sn 리스트 (이 중에서만 검색)
            metadata_filter: Pinecone 메타데이터 필터 (topic별로 다름)

        Returns:
            [{"id": ..., "score": ..., "mb_sn": ..., "index": ..., "topic": ..., "text": ...}]
        """
        # top_k 유효성 검사
        if top_k <= 0:
            return []

        # 후보 mb_sn이 비어있는 경우 처리
        if filter_mb_sns is not None and len(filter_mb_sns) == 0:
            return []

        # 카테고리에 해당하는 Pinecone topic 가져오기
        pinecone_topic = self.category_config.get(category, {}).get("pinecone_topic", category)

        # 기본 필터: topic
        filter_dict = {"topic": pinecone_topic}

        # mb_sn 필터 추가 (이전 단계에서 선별된 mb_sn들로 제한)
        if filter_mb_sns:
            filter_dict["mb_sn"] = {"$in": filter_mb_sns}

        # 1차 시도: 메타데이터 필터 적용
        if metadata_filter:
            filter_with_metadata = filter_dict.copy()
            # 리스트 값을 $in 연산자로 변환
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    # 리스트인 경우 $in 연산자 사용
                    filter_with_metadata[key] = {"$in": value}
                elif isinstance(value, dict):
                    # 이미 Pinecone 필터 형식인 경우 (예: {"$lte": 300})
                    filter_with_metadata[key] = value
                else:
                    # 단일 값인 경우 그대로 사용
                    filter_with_metadata[key] = value

            # Pinecone 검색 (메타데이터 필터 포함)
            # top_k를 그대로 사용 (제한 없음)
            try:
                search_results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_with_metadata
                )

                # 무응답 필터링 제거 (모든 결과 포함)
                valid_results = list(search_results.matches)

                # 🔄 Fallback: 결과가 0개면 메타데이터 필터 없이 재검색
                if len(valid_results) == 0:
                    search_results = self.index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter_dict  # 메타데이터 필터 제거
                    )
                    valid_results = list(search_results.matches)
            except Exception as e:
                logger.warning(f"Pinecone 검색 오류 (메타데이터 필터): {e}, Fallback 시도")
                # Fallback: 메타데이터 필터 없이 재검색
                search_results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                #  : 무응답 필터링 제거
                valid_results = list(search_results.matches)
        else:
            # 메타데이터 필터 없이 검색
            # top_k를 그대로 사용 (제한 없음)
            try:
                search_results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                #  : 무응답 필터링 제거
                valid_results = list(search_results.matches)
            except Exception as e:
                logger.error(f"Pinecone 검색 오류: {e}")
                return []

        # Pinecone이 이미 정렬된 결과를 그대로 사용 (재정렬하지 않음)
        # 결과 변환 (상위 top_k개만)
        matches = []
        for match in valid_results[:top_k]:
            metadata = match.metadata or {}
            matches.append({
                "id": match.id,
                "score": match.score,
                "mb_sn": metadata.get("mb_sn", ""),
                "index": metadata.get("index", 0),
                "topic": metadata.get("topic", ""),
                "text": metadata.get("text", ""),
                "지역": metadata.get("지역", ""),
                "연령대": metadata.get("연령대", ""),
                "성별": metadata.get("성별", "")
            })

        return matches

