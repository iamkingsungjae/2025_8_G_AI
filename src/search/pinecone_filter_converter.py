"""Pinecone 필터 변환기 (프론트엔드 필터 → Pinecone 필터)"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PineconeFilterConverter:
    """프론트엔드 필터를 Pinecone 메타데이터 필터로 변환"""

    @staticmethod
    def convert_to_pinecone_filters(filters_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        프론트엔드 필터를 Pinecone 카테고리별 필터로 변환
        
        Args:
            filters_dict: 프론트엔드 필터 딕셔너리
                - selectedGenders: ["남", "여"]
                - selectedRegions: ["서울", "경기"]
                - ageRange: [20, 40]
                - selectedIncomes: ["300~500만원"]
                - quickpollOnly: true/false
        
        Returns:
            카테고리별 Pinecone 필터
            {
                "기본정보": {"지역": {"$in": [...]}, "성별": {...}, ...},
                "직업소득": {"개인소득_min": {...}, ...}
            }
        """
        category_filters = {}
        
        # 기본정보 필터 (인구 topic)
        basic_filters = {}
        
        # 성별 필터
        if selected_genders := filters_dict.get("selectedGenders"):
            if isinstance(selected_genders, list) and len(selected_genders) > 0:
                # '남'/'여' 정규화
                normalized_genders = []
                for gender in selected_genders:
                    if gender in ['남', '남성', 'M', 'male']:
                        normalized_genders.append('남')
                    elif gender in ['여', '여성', 'F', 'female']:
                        normalized_genders.append('여')
                    else:
                        normalized_genders.append(str(gender))
                
                if len(normalized_genders) == 1:
                    basic_filters["성별"] = normalized_genders[0]
                else:
                    basic_filters["성별"] = {"$in": normalized_genders}
        
        # 지역 필터
        if selected_regions := filters_dict.get("selectedRegions"):
            if isinstance(selected_regions, list) and len(selected_regions) > 0:
                # 지역명 정규화
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
                }
                
                normalized_regions = [region_mapping.get(r, r) for r in selected_regions]
                
                if len(normalized_regions) == 1:
                    basic_filters["지역"] = normalized_regions[0]
                else:
                    basic_filters["지역"] = {"$in": normalized_regions}
        
        # 나이 필터 → 연령대 변환
        if age_range := filters_dict.get("ageRange"):
            if isinstance(age_range, list) and len(age_range) == 2:
                age_min, age_max = age_range[0], age_range[1]
                
                # 연령대 계산
                age_groups = []
                if age_min is not None:
                    min_group = (age_min // 10) * 10
                else:
                    min_group = 0
                
                if age_max is not None:
                    max_group = (age_max // 10) * 10
                else:
                    max_group = 100
                
                # 연령대 리스트 생성
                for age in range(min_group, min(max_group + 1, 100), 10):
                    age_groups.append(f"{age}대")
                
                if age_groups:
                    if len(age_groups) == 1:
                        basic_filters["연령대"] = age_groups[0]
                    else:
                        basic_filters["연령대"] = {"$in": age_groups}
                
                # 나이 범위 필터링 (Pinecone은 $and를 지원하지 않으므로 연령대만 사용)
                # 나이 필터는 연령대 필터로 충분히 처리됨
        
        if basic_filters:
            category_filters["기본정보"] = basic_filters
        
        # 직업소득 필터
        income_filters = {}
        
        if selected_incomes := filters_dict.get("selectedIncomes"):
            if isinstance(selected_incomes, list) and len(selected_incomes) > 0:
                # 소득 범위 파싱
                for income_range_str in selected_incomes:
                    if "~" in income_range_str:
                        # "300~500만원" 형식 파싱
                        parts = income_range_str.replace("~", " ").replace("만원", "").split()
                        if len(parts) == 2:
                            try:
                                min_income = int(parts[0])
                                max_income = int(parts[1])
                                
                                # 개인소득 필터
                                if "개인소득_min" not in income_filters:
                                    income_filters["개인소득_min"] = {"$lte": max_income}
                                    income_filters["개인소득_max"] = {"$gte": min_income}
                                else:
                                    # 여러 범위가 있으면 OR 조건 (복잡하므로 첫 번째만 사용)
                                    pass
                            except (ValueError, TypeError):
                                pass
        
        if income_filters:
            category_filters["직업소득"] = income_filters
        
        # 퀵폴 응답 보유만 보기 필터
        if quickpoll_only := filters_dict.get("quickpollOnly"):
            if quickpoll_only is True:
                # coverage 필드가 "qw"인 패널만 필터링
                if "기본정보" not in category_filters:
                    category_filters["기본정보"] = {}
                # Pinecone 메타데이터에 coverage 필드가 있다고 가정
                # 실제 필드명은 데이터베이스 스키마에 따라 조정 필요
                category_filters["기본정보"]["coverage"] = "qw"
        
        return category_filters

