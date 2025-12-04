"""Pinecone 결과 필터"""
from typing import Dict, List, Any
from collections import OrderedDict
import logging
import time

logger = logging.getLogger(__name__)


class PineconeResultFilter:
    """카테고리 순서에 따라 단계적으로 mb_sn을 필터링 (Pinecone 최적화)"""

    def __init__(self, pinecone_searcher):
        self.searcher = pinecone_searcher

    def filter_by_categories(
        self,
        embeddings: Dict[str, List[float]],
        category_order: List[str],
        final_count: int = None,
        topic_filters: Dict[str, Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        카테고리 순서대로 단계적으로 필터링하여 최종 mb_sn 리스트 반환

        Args:
            embeddings: {"카테고리명": [임베딩 벡터]}
            category_order: 카테고리 순서 (예: ["기본정보", "직업소득", "자동차"])
            final_count: 최종 출력할 mb_sn 개수 (None이면 조건 만족하는 전체 반환)
            topic_filters: topic별 메타데이터 필터 (예: {"기본정보": {...}, "직업소득": {...}})

        Returns:
            최종 선별된 mb_sn 리스트
        """
        if not category_order:
            return []

        filter_start = time.time()

        # 첫 번째 카테고리로 초기 선별
        first_category = category_order[0]
        first_embedding = embeddings.get(first_category)

        if first_embedding is None:
            return []

        # 첫 번째 카테고리의 메타데이터 필터 가져오기
        first_filter = (topic_filters or {}).get(first_category, {})
        has_metadata_filter = bool(first_filter)


        # 초기 검색 수 결정
        if final_count is None:
            # 명수 미명시
            if has_metadata_filter:
                initial_count = 10000
            else:
                initial_count = 10000
        else:
            # 명수 명시됨
            if has_metadata_filter:
                initial_count = 10000
            else:
                initial_count = max(final_count * 10, 2000)

        first_results = self.searcher.search_by_category(
            query_embedding=first_embedding,
            category=first_category,
            top_k=initial_count,
            filter_mb_sns=None,  # 첫 단계는 전체 검색
            metadata_filter=first_filter
        )

        # 메타데이터 필터 사용 시 - 필터 조건 만족하는 패널 중 유사도 높은 순으로 정렬
        if has_metadata_filter:
            # 필터 조건을 만족하는 패널의 유사도 점수 수집
            filtered_mb_sn_scores = {}
            for r in first_results:
                mb_sn = r.get("mb_sn", "")
                if mb_sn:
                    score = r.get("score", 0.0)
                    # 최고 점수만 유지 (여러 카테고리에서 같은 mb_sn이 나올 수 있음)
                    if mb_sn not in filtered_mb_sn_scores or score > filtered_mb_sn_scores[mb_sn]:
                        filtered_mb_sn_scores[mb_sn] = score
            
            #  유사도 점수 기준으로 정렬 (필터 조건 만족하는 패널 중에서)
            sorted_filtered = sorted(
                filtered_mb_sn_scores.items(), 
                key=lambda x: x[1], 
                reverse=True  # 높은 점수부터
            )
            
            # 필터가 있을 때는 전체 유지 (조기 제한 없음)
            candidate_mb_sns = [mb_sn for mb_sn, score in sorted_filtered]
            
        else:
            # 필터 없을 때
            # 정렬 순서 유지하며 후보군 구성
            first_sorted = sorted(
                [r for r in first_results if r.get("mb_sn")],
                key=lambda x: x["score"],
                reverse=True
            )
            candidate_mb_sns = list(OrderedDict.fromkeys(r["mb_sn"] for r in first_sorted))

            if final_count is not None and not has_metadata_filter:
                candidate_mb_sns = candidate_mb_sns[:max(final_count * 10, 10000)]

        # 후보가 없으면 빈 리스트 반환
        if len(candidate_mb_sns) == 0:
            return []

        # 나머지 카테고리로 점진적 필터링
        for i, category in enumerate(category_order[1:], start=2):
            embedding = embeddings.get(category)

            if embedding is None:
                continue

            # 현재 카테고리의 메타데이터 필터 가져오기
            category_filter = (topic_filters or {}).get(category, {})
            has_category_filter = bool(category_filter)


            # 후보가 비어있으면 필터링 중단
            if len(candidate_mb_sns) == 0:
                break

            # 후보 수에 따라 검색 수 결정
            if final_count is None and has_category_filter:
                # 명수 미명시 + 메타데이터 필터 O → 충분히 큰 수
                search_count = min(len(candidate_mb_sns) * 3, 10000)
            else:
                # 명수 명시 or 필터 없음 → 적당히
                search_count = min(len(candidate_mb_sns) * 2, 10000)

            search_count = max(search_count, 1)

            results = self.searcher.search_by_category(
                query_embedding=embedding,
                category=category,
                top_k=search_count,
                filter_mb_sns=candidate_mb_sns,  # 이전 단계에서 선별된 mb_sn들로 제한
                metadata_filter=category_filter
            )

            # 메타데이터 필터 여부에 따라 다른 전략
            if has_category_filter:
                # 메타데이터 필터 O → 필터 조건 만족하는 패널 중 유사도 높은 순으로 정렬
                filtered_mb_sns = set([r["mb_sn"] for r in results if r.get("mb_sn") in candidate_mb_sns])
                
                # mb_sn별 최고 점수로 정렬 (여러 카테고리에서 같은 mb_sn이 나올 수 있음)
                mb_sn_scores = {}
                for r in results:
                    mb_sn = r.get("mb_sn", "")
                    if mb_sn in filtered_mb_sns:
                        score = r.get("score", 0.0)
                        if mb_sn not in mb_sn_scores or score > mb_sn_scores[mb_sn]:
                            mb_sn_scores[mb_sn] = score
                
                # 유사도 점수 기준으로 정렬 (필터 조건 만족하는 패널 중에서)
                sorted_mb_sns = sorted(mb_sn_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 필터가 있을 때는 전체 유지 (조기 제한 없음)
                candidate_mb_sns = [mb_sn for mb_sn, score in sorted_mb_sns]
                
            else:
                # 메타데이터 필터 X → 벡터 유사도 기반 상위 선별
                mb_sn_scores = {}
                for r in results:
                    mb_sn = r.get("mb_sn", "")
                    if mb_sn in candidate_mb_sns:
                        if mb_sn not in mb_sn_scores or r.get("score", 0.0) > mb_sn_scores[mb_sn]:
                            mb_sn_scores[mb_sn] = r.get("score", 0.0)

                sorted_mb_sns = sorted(mb_sn_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 다음 단계를 위한 후보 수 결정
                if final_count is None:
                    # 명수 미명시 → 전체 유지
                    next_candidate_count = len(sorted_mb_sns)
                else:
                    # 명수 명시 → 여유있게, 최소 10000개 보장
                    next_candidate_count = max(final_count * 3, 10000)
                
                candidate_mb_sns = [mb_sn for mb_sn, score in sorted_mb_sns[:next_candidate_count]]

        # 최종 결과도 score 정렬 보장 (마지막 카테고리 점수만 사용)
        final_results = self.searcher.search_by_category(
            query_embedding=embeddings[category_order[-1]],
            category=category_order[-1],
            top_k=len(candidate_mb_sns),
            filter_mb_sns=candidate_mb_sns
        )

        final_scores = {}
        for r in final_results:
            mb_sn = r.get("mb_sn", "")
            if mb_sn in candidate_mb_sns:
                score = r.get("score", 0.0)
                # 최고 점수만 유지 (여러 카테고리에서 같은 mb_sn이 나올 수 있음)
                if mb_sn not in final_scores or score > final_scores[mb_sn]:
                    final_scores[mb_sn] = score

        final_sorted = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_mb_sns = [mb_sn for mb_sn, score in final_sorted]
        
        if final_count is not None:
            final_mb_sns = final_mb_sns[:final_count]
            logger.info(
                f"최종 {len(final_mb_sns)}개 패널 선별 완료 ({final_count}명 요청)"
            )
        else:
            logger.info(
                f"최종 {len(final_mb_sns)}개 패널 선별 완료 (조건 만족하는 전체 반환)"
            )

        final_results = [{"mb_sn": mb_sn, "score": final_scores.get(mb_sn, 0.0)} for mb_sn in final_mb_sns]

        return final_results

