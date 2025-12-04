"""카테고리별 메타데이터 필터 추출기"""
from typing import Dict, Any
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class MetadataFilterExtractor:
    """카테고리별 메타데이터 필터 추출 및 정규화 (복수 값 지원) - LLM 기반"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Anthropic API 키
        """
        if not api_key:
            logger.error("[MetadataFilterExtractor] API 키가 비어있습니다!")
        elif len(api_key) < 50:
            logger.warning(f"[MetadataFilterExtractor] API 키가 너무 짧습니다 (길이: {len(api_key)})")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"

    def extract_filters(self, metadata: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        특정 카테고리에 적용할 메타데이터 필터를 추출 및 정규화
        
        Args:
            metadata: 전체 메타데이터
            category: 카테고리명 (예: "기본정보")
        
        Returns:
            정규화된 메타데이터 필터 (복수 값 포함)
            예: {"지역": ["서울", "경기"], "연령대": ["10대", "20대"], "성별": "남", "결혼여부": "기혼"}
        """
        # Pinecone 실제 메타데이터 구조에 맞춘 카테고리별 매핑
        # Pinecone 확인 결과:
        #   - "인구" topic: 지역, 지역구, 연령대, 성별, 나이, 결혼여부, 자녀수, 가족수, 학력 (9개 필드)
        #   - 기타 모든 topic: topic, index, mb_sn만 존재 (메타데이터 필터 사용 불가)
        CATEGORY_METADATA_MAPPING = {
            "기본정보": ["지역", "지역구", "연령대", "성별", "나이", "결혼여부", "자녀수", "가족수", "학력"],
            "직업소득": ["개인소득", "가구소득"],  # Pinecone에서 소득 필터 지원
        }
        
        applicable_keys = CATEGORY_METADATA_MAPPING.get(category, [])
        
        if not applicable_keys:
            return {}
        
        # 해당 카테고리에 적용 가능한 메타데이터만 추출
        relevant_metadata = {}
        for key in applicable_keys:
            if key in metadata:
                relevant_metadata[key] = metadata[key]
        
        if not relevant_metadata:
            return {}
        
        # 복수 값 보존을 위해 rule-based 정규화 직접 사용
        # LLM이 리스트를 단일 값으로 변환하는 문제를 해결하기 위해 rule-based 사용
        normalized_filter = self._rule_based_normalize(relevant_metadata)
        return normalized_filter

    def _rule_based_normalize(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """규칙 기반 정규화 (복수 값 지원, 새 필터 포함)"""
        filter_dict = {}
        
        # 지역명 매핑
        region_mapping = {
            "서울특별시": "서울", "서울시": "서울",
            "부산광역시": "부산", "부산시": "부산",
            "대구광역시": "대구", "대구시": "대구",
            "인천광역시": "인천", "인천시": "인천",
            "광주광역시": "광주", "광주시": "광주",
            "대전광역시": "대전", "대전시": "대전",
            "울산광역시": "울산", "울산시": "울산",
            "세종특별자치시": "세종", "세종시": "세종",
            "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원",
            "충청북도": "충북", "충북도": "충북",
            "충청남도": "충남", "충남도": "충남",
            "전라북도": "전북", "전북도": "전북", "전북특별자치도": "전북",
            "전라남도": "전남", "전남도": "전남",
            "경상북도": "경북", "경북도": "경북",
            "경상남도": "경남", "경남도": "경남",
            "제주특별자치도": "제주", "제주도": "제주", "제주시": "제주",
            "해외": "해외", "외국": "해외", "국외": "해외",
        }
        
        # 학력 매핑 (텍스트 정규화)
        education_mapping = {
            "고졸": "고등학교 졸업 이하",
            "고등학교": "고등학교 졸업 이하",
            "고등학교 졸업": "고등학교 졸업 이하",
            "대학생": "대학교 재학",
            "대학 재학": "대학교 재학",
            "대학교 재학": "대학교 재학",
            "대재": "대학교 재학",
            "대졸": "대학교 졸업",
            "대학 졸업": "대학교 졸업",
            "대학교 졸업": "대학교 졸업",
            "대학원": "대학원 재학/졸업 이상",
            "석사": "대학원 재학/졸업 이상",
            "박사": "대학원 재학/졸업 이상",
            "대학원 재학": "대학원 재학/졸업 이상",
            "대학원 졸업": "대학원 재학/졸업 이상",
        }
        
        for key, value in metadata.items():
            if not value or value == '':
                continue
            
            # 리스트인 경우 모든 값을 정규화
            if isinstance(value, list):
                normalized_list = []
                for item in value:
                    if key == "지역":
                        normalized_list.append(region_mapping.get(item, item))
                    elif key == "성별":
                        if item in ["남성", "남자", "male", "M"]:
                            normalized_list.append("남")
                        elif item in ["여성", "여자", "female", "F"]:
                            normalized_list.append("여")
                        else:
                            normalized_list.append(item)
                    elif key == "학력":
                        normalized_list.append(education_mapping.get(item, item))
                    else:
                        normalized_list.append(item)
                filter_dict[key] = normalized_list
            else:
                # 단일 값인 경우
                if key == "지역":
                    value = region_mapping.get(value, value)
                elif key == "성별":
                    if value in ["남성", "남자", "male", "M"]:
                        value = "남"
                    elif value in ["여성", "여자", "female", "F"]:
                        value = "여"
                elif key == "학력":
                    value = education_mapping.get(value, value)
                elif key == "결혼여부":
                    # 결혼여부 정규화: "기혼", "미혼", "기타" 중 하나
                    if value in ["결혼", "결혼한", "기혼자"]:
                        value = "기혼"
                    elif value in ["미혼자", "결혼 안한"]:
                        value = "미혼"
                # 나이, 자녀수, 가족수는 숫자 그대로 유지
                elif key in ["나이", "자녀수", "가족수"]:
                    # 문자열이면 int로 변환 시도
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                # 개인소득, 가구소득 범위 비교 로직 (Pinecone 필터 형식)
                elif key in ["개인소득", "가구소득"]:
                    import re
                    # value는 쿼리에서 추출된 값 (예: 300만원)
                    # Pinecone 필터: {key_min: {$lte: 300}, key_max: {$gte: 300}}
                    if isinstance(value, (int, float)):
                        filter_dict[f"{key}_min"] = {"$lte": value}
                        filter_dict[f"{key}_max"] = {"$gte": value}
                        continue  # filter_dict[key] = value 실행 방지
                    elif isinstance(value, str):
                        # "300만원", "300" 등 처리
                        match = re.search(r'(\d+)', str(value))
                        if match:
                            value_int = int(match.group(1))
                            filter_dict[f"{key}_min"] = {"$lte": value_int}
                            filter_dict[f"{key}_max"] = {"$gte": value_int}
                            continue
                
                filter_dict[key] = value
        
        return filter_dict


