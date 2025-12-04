"""라이프스타일 분류 서비스 (Final_panel_insight.ipynb 기반)"""
from typing import Dict, Any, List, Optional
import json
import re
import logging
import numpy as np
from anthropic import Anthropic
from pinecone import Pinecone

from app.core.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, load_category_config

logger = logging.getLogger(__name__)

# 라이프스타일 정의 (Final_panel_insight.ipynb에서 가져옴)
LIFESTYLE_DEFINITIONS = {
    "1": {
        "name": "실용·효율 중심 도시형",
        "definition": "합리적인 소비와 기능 중심의 실용성을 중시하며, 낭비 없는 생활과 체계적인 루틴을 통해 삶의 효율을 높이려는 성향을 가진 사람.",
        "core_signals": [
            "가격 대비 가치, 효율성에 대한 관심",
            "생필품 중심의 계획적 소비",
            "할인·포인트 등 실속 있는 소비 혜택 선호",
            "아침 알람 설정, 혼밥 등 시간 관리 중심 생활"
        ]
    },
    "2": {
        "name": "디지털·AI 친화형",
        "definition": "AI와 디지털 기술을 일상에 자연스럽게 통합하며, 새로운 도구에 대한 호기심과 적응력이 높은 사람. 기술을 통해 문제를 해결하고, 다양한 플랫폼을 능숙하게 활용한다.",
        "core_signals": [
            "챗GPT 등 AI 서비스 사용 경험 및 선호",
            "앱 및 디지털 서비스 사용 빈도 높음",
            "새로운 기술·앱에 대한 개방성"
        ]
    },
    "3": {
        "name": "건강·체력관리형",
        "definition": "운동, 식습관, 수면 등 자기 몸 관리에 관심이 높으며, 지속적인 체력 유지를 삶의 중요한 요소로 생각하는 사람.",
        "core_signals": [
            "유산소·헬스 등 체력 활동 습관",
            "야식·간식 조절 등 식습관 관리",
            "수면 루틴 및 컨디션 조절에 민감함"
        ]
    },
    "4": {
        "name": "감정·힐링 중심형",
        "definition": "정서적 안정과 스트레스 해소를 중요하게 여기며, 휴식이나 감정 기반의 소비를 통해 마음의 만족을 추구하는 사람.",
        "core_signals": [
            "스트레스 원인 및 해소 방법에 대한 민감도",
            "본인을 위한 소비(감정적 만족) 선호",
            "힐링 경험(선물, 여유, 안정)에 가치를 둠"
        ]
    },
    "5": {
        "name": "뷰티·자기관리형",
        "definition": "외모, 피부, 패션에 대한 꾸준한 관심과 투자를 통해 자기 표현과 자기 효능감을 중요하게 여기는 사람.",
        "core_signals": [
            "피부 상태에 대한 민감도",
            "스킨케어 소비 루틴 고정",
            "여름철 뷰티·패션 필수템 보유"
        ]
    },
    "6": {
        "name": "환경·정리·미니멀형",
        "definition": "불필요한 소비를 줄이고, 환경 보호와 물건 정리에 집중하는 삶의 방식을 추구하는 사람. 재활용, 절제된 소비를 일상에서 실천한다.",
        "core_signals": [
            "일회용품 사용 줄이기 실천",
            "미니멀 vs 맥시멀 인식 명확",
            "버리기 아까운 물건의 처리 방식 고민"
        ]
    },
    "7": {
        "name": "여가·경험 중심형",
        "definition": "여행, 물놀이, 레저 등 활동 기반의 즐거움을 중시하며, 즉흥적이고 유연한 라이프스타일을 추구하는 사람.",
        "core_signals": [
            "여행 스타일, 장소 선호 명확",
            "레저 중심 소비·경험 중요시",
            "여름철 외부 활동에 대한 기대감"
        ]
    },
    "8": {
        "name": "가족·관계 중심형",
        "definition": "가족, 반려동물, 유대 관계를 삶의 중심으로 여기며, 감정적 연결과 돌봄에 높은 가치를 두는 사람.",
        "core_signals": [
            "반려동물과의 생활 경험",
            "가족을 위한 선택과 소비",
            "행복한 노년·관계에 대한 관심"
        ]
            }
            }


def load_texts_by_mb_sn(panel_id: str, max_results: int = 200) -> List[Dict[str, str]]:
    """
    Pinecone에서 특정 mb_sn의 모든 feature_sentence를 가져옴 (Final_panel_insight.ipynb 기반)
    
    Args:
        panel_id: 패널 ID (mb_sn)
        max_results: 최대 결과 수 (기본값 200)
        
    Returns:
        [{"topic": "...", "text": "..."}] 형태의 리스트
    """
    try:
        category_config = load_category_config()
        
        # Pinecone 인덱스 연결
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # 인덱스 차원 동적으로 가져오기
        stats = index.describe_index_stats()
        dimension = stats.get('dimension', 1536)
        
        # 랜덤 벡터로 검색 (메타데이터만 필요)
        random_vector = np.random.rand(dimension).astype(np.float32).tolist()
        norm = np.linalg.norm(random_vector)
        if norm > 0:
            random_vector = (np.array(random_vector) / norm).tolist()
        
        # 모든 topic에서 mb_sn으로 검색
        all_features = []
        
        # 각 카테고리의 topic으로 검색
        for category_name, category_info in category_config.items():
            pinecone_topic = category_info.get("pinecone_topic")
            if not pinecone_topic:
                continue
            
            try:
                results = index.query(
                    vector=random_vector,
                    top_k=max_results,
                    include_metadata=True,
                    filter={
                        "topic": pinecone_topic,
                        "mb_sn": panel_id
                    }
                )
                
                for match in results.matches:
                    metadata = match.metadata or {}
                    topic = metadata.get("topic", pinecone_topic)
                    text = metadata.get("text", "")
                    if text:
                        all_features.append({
                            "topic": topic,
                            "text": text
                        })
            except Exception as e:
                logger.debug(f"[load_texts_by_mb_sn] {category_name} topic 검색 실패: {e}")
                continue
        
        logger.info(f"[load_texts_by_mb_sn] {panel_id}: {len(all_features)}개 feature_sentence 수집")
        return all_features
        
    except Exception as e:
        logger.error(f"[load_texts_by_mb_sn] 실패: {panel_id}, 오류: {e}", exc_info=True)
        return []


def generate_lifestyle_summary(panel_id: str, api_key: str) -> str:
    """
    패널 ID를 기반으로 라이프스타일 분류 후 요약 텍스트 생성
    
    Args:
        panel_id: 패널 ID (mb_sn)
        api_key: Anthropic API 키
        
    Returns:
        AI 요약 텍스트
    """
    try:
        # 1. Pinecone에서 feature_sentence 수집
        feature_list = load_texts_by_mb_sn(panel_id)
        
        if not feature_list or len(feature_list) <= 1:
            return "라이프스타일을 분류할 수 있는 충분한 정보가 없습니다."
        
        # 2. 라이프스타일 분류
        classifier = LifestyleClassifier(api_key)
        result = classifier.classify(feature_list)
        
        # 3. 요약 텍스트 생성
        if not result.get("lifestyle") or len(result["lifestyle"]) == 0:
            return result.get("message", "라이프스타일을 분류할 수 없습니다.")
        
        # primary 라이프스타일로 요약 생성
        primary = None
        secondary = None
        for lifestyle in result["lifestyle"]:
            if lifestyle.get("role") == "primary":
                primary = lifestyle
            elif lifestyle.get("role") == "secondary":
                secondary = lifestyle
        
        summary_parts = []
        if primary:
            summary_parts.append(f"{primary.get('lifestyle_name', '')} 유형으로, {primary.get('reason', '')}")
        if secondary:
            summary_parts.append(f"또한 {secondary.get('lifestyle_name', '')} 특성도 보이며, {secondary.get('reason', '')}")
        
        summary = " ".join(summary_parts)
        if not summary:
            summary = "라이프스타일 분석이 완료되었습니다."
        
        return summary
        
    except Exception as e:
        logger.error(f"[generate_lifestyle_summary] 실패: {panel_id}, 오류: {e}", exc_info=True)
        return "라이프스타일 요약 생성 중 오류가 발생했습니다."


def normalize_feature_data(feature_data_raw: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Pinecone에서 가져온 feature_sentence 리스트를 normalize.
    이미 topic/text 형태이므로 그대로 반환.
    """
    if not isinstance(feature_data_raw, list):
        return []

    normalized = []
    for item in feature_data_raw:
        topic = item.get("topic")
        text = item.get("text")
        if topic and text:
            normalized.append({
                "topic": topic,
                "text": text
            })

    return normalized


def extract_json_from_response(text: str) -> str:
    """LLM 응답에서 JSON 추출"""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


class LifestyleClassifier:
    """라이프스타일 분류기 (Final_panel_insight.ipynb 기반)"""
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Anthropic API 키
        """
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-opus-4-1-20250805"
        logger.debug(f"[LifestyleClassifier] 초기화 완료, 모델: {self.model}")
    
    def classify(self, feature_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        feature_list를 기반으로 라이프스타일 분류
        
        Args:
            feature_list: [{"topic": "...", "text": "..."}] 형태의 리스트
            
        Returns:
            {
                "lifestyle": [...],
                "evidence_topics": [...]
            }
        """
        if not feature_list or len(feature_list) <= 1:
            return {
                "lifestyle": [],
                "message": "라이프스타일을 분류할 수 없습니다.",
                "evidence_topics": []
            }
        
        # feature_list -> topic/text 구조로 정규화
        analysis_items = [
            {"topic": f["topic"], "text": f["text"]}
            for f in feature_list
        ]

        lifestyle_json = json.dumps(LIFESTYLE_DEFINITIONS, ensure_ascii=False, indent=2)
        analysis_json = json.dumps(analysis_items, ensure_ascii=False, indent=2)

        # 프롬프트 생성 (전체 규칙 포함)
        prompt = self._build_prompt(lifestyle_json, analysis_json)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0  # 60초 타임아웃 (라이프스타일 분류는 더 복잡하므로)
            )
            
            text = response.content[0].text
            logger.debug(f"[라이프스타일 분류] LLM 원본 응답: {text[:500]}")
            
            # JSON 추출
            json_str = extract_json_from_response(text)
            if not json_str:
                logger.warning("[라이프스타일 분류] JSON 추출 실패")
                return {
                    "lifestyle": [],
                    "message": "라이프스타일 분류 실패",
                    "evidence_topics": []
                }
            
            parsed = json.loads(json_str)
            logger.info(f"[라이프스타일 분류] 분류 완료: {parsed}")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"[라이프스타일 분류] JSON 파싱 실패: {e}")
            return {
                "lifestyle": [],
                "message": "라이프스타일 분류 실패",
                "evidence_topics": []
            }
        except Exception as e:
            logger.error(f"[라이프스타일 분류] 분류 실패: {e}", exc_info=True)
            return {
                "lifestyle": [],
                "message": "라이프스타일 분류 실패",
                "evidence_topics": []
            }
    
    def _build_prompt(self, lifestyle_json: str, analysis_json: str) -> str:
        """프롬프트 생성 (Final_panel_insight.ipynb의 전체 규칙 포함)"""
        return f"""
당신은 사람의 라이프스타일을 정밀하게 해석하는 전문 프로파일러(Personality & Lifestyle Analyst)입니다.
당신의 임무는 입력된 feature_sentence들을 기반으로 이 사람이 어떤 라이프스타일 **행동 유형**에 해당하는지를 논리적·일관적으로 분류하는 것입니다.

입력은 한 사람의 행동 패턴, 소비 성향, 감정적 반응, 여가·취향, 디지털 태도, 가치관을 보여주는 문장 목록입니다.
각 문장은 설문 기반 원자료를 정규화한 것으로, 하나의 행동적 신호 또는 가치관을 나타냅니다.

────────────────────────────────────────
[0] 참고해야 하는 행동 유형 정의 (definition + core_signals)
────────────────────────────────────────

아래 JSON은 각 행동 유형의 의미적 정의, 가치관 특징, 행동 패턴 신호입니다.

매우 중요:
lifestyle_definitions(JSON)는 "참고용" 정의 자료이며, 절대로 단독 기준이 되어서는 안 됩니다.

행동 유형 선택 시 최우선 기준은 다음 두 가지입니다:
1) [2] 해석 원칙
2) [3]~[10] 각 유형 판별 기준 및 금지 규칙

lifestyle_definitions(JSON)는 위 규칙들로 후보를 좁힌 뒤,
최종적으로 표현이 맞는지를 확인하는 보조 참고 자료로만 사용하세요.

절대로 lifestyle_definitions(JSON)에 등장하는 단어만으로 유형을 선택하지 마세요.
전체 feature_sentence들의 의미 패턴이 우선입니다.

행동 유형 정의(JSON):
{lifestyle_json}

────────────────────────────────────────
분석 대상 feature_sentence 목록
────────────────────────────────────────
아래 JSON 배열은 한 사람에 대한 topic + text 정보입니다.
이 목록의 **모든 feature_sentence**를 끝까지 읽고,
규칙에 등장하지 않는 내용이라도 반드시 전체 맥락 해석에 포함해야 합니다.

{analysis_json}

────────────────────────────────────────
[1] 행동 유형 목록 (최대 2개만 선택 가능)
────────────────────────────────────────
선택할 수 있는 행동 유형은 아래 8개입니다.

1. 실용·효율 중심 도시형
2. 디지털·AI 친화형
3. 건강·체력관리형
4. 감정·힐링 중심형
5. 뷰티·자기관리형
6. 환경·정리·미니멀형
7. 여가·경험 중심형
8. 가족·관계 중심형

출력 규칙:
- feature_sentence 개수가 1개 이하이면 → 어떤 유형도 선택하지 말고 "행동 유형을 분류할 수 없습니다!"만 출력
- 가장 일관된 유형이 단 하나라면 → primary 1개만 출력
- 단정 짓기 어려울 때만 → primary + secondary (최대 2개)
- secondary는 정말 필요할 때만 포함
- 3개 이상은 절대 금지

────────────────────────────────────────
[2] 해석 원칙 
────────────────────────────────────────

 가치관 신호  
- 이 사람이 무엇을 중요하게 여기는가  
- 행동의 동기·감정·선호  
- 삶의 만족 지점  

 행동의 이유 중심  
무엇을 했는가 보다 "왜 그렇게 행동하는가"를 중심으로 해석하세요.

 반복성·우선성  
한두 문장이 아닌 **반복되는 패턴**을 우선으로 판단하세요.

 강화 신호 단일 출현 금지(모든 유형 공통)
- 특정 유형과 관련된 강화 신호가 "1개만" 등장한 경우  
  → 해당 유형은 primary, secondary 모두 선택 금지.
- 후보군으로도 올리면 안 됨.
- 강화 신호는 반드시 2개 이상 반복적으로 등장할 때만 의미가 있다고 간주.
- 반복되지 않는 단일 신호(예: 운동 1회, 여행 1회, 스킨케어 1회 등)는  
  '단편적/우연적 행동'으로 처리하고 유형 근거로 사용하면 안 됨.

 단편 신호 배제  
- 빠른 배송  
- 우산 구매  
- 포인트 적립  
- 일회용 소비 등 단발적 행동은 근거 금지

 전체 feature 기반 해석  
규칙에 직접 언급되지 않는 feature라도 반드시 전체 패턴 속에서 의미를 해석해야 합니다.

────────────────────────────────────────
[3] 실용·효율 중심 도시형(1번)  — QPOLL 기반 강화
────────────────────────────────────────
아래 중 하나라도 있으면 실용형(1번) 선택 금지:
- 여행·레저·경험 기반 활동이 2개 이상  
- 감정 기반 소비·기분전환 소비 반복  
- 반려동물·사진·취향 중심 행동 반복  
- 여가 활동이 2회 이상 반복  
- 정서·취향 기반 만족 신호 반복  
- 스스로 맥시멀리스트라고 밝힘  

※ QPOLL 기반 금지 신호
- 풍경/여행/반려동물 사진 반복 → 실용형 근거 금지  
- OTT 서비스 개수, 앱 사용 종류(SNS·쇼핑 등) → 실용형 근거 금지  
- 빠른 배송 1회 → 근거 금지  

실용형 강화 신호(QPOLL 기반):
- 설 선물: 생필품/현금/건강식품 반복 선택  
- 소비 기준이 가격·효율·필요성 중심  
- 포인트·캐시백·멤버십을 꼼꼼하게 챙김  
- 중고 판매/재활용/기부를 "낭비 방지" 목적 반복  
- 여행 스타일: 계획형  
- 여러 개 알람 설정, 루틴 기반 생활  

실용형 primary 조건:
- 절약/효율/계획성 신호 2회 이상  
- 감정·여가·취향 신호 1회 이하  


────────────────────────────────────────
[4] 디지털·AI 친화형(2번) — 조정된 우선순위 규칙
────────────────────────────────────────
강한 신호(반복 필요):
- ChatGPT, 딥시크, Gemini, Claude 등 AI 기반 서비스의 "반복적·능동적 사용"
- "주로 사용", "선호", "자주 사용"처럼 명확한 태도 표현
- 문서작성/번역/코딩/이미지 생성과 같은 능동적 활용
- 정보 탐색/학습/업무 흐름에서 AI 사용이 반복적으로 등장

약화/금지 신호:
- '사용해본 적 있다'는 단발성 경험 → 근거 금지
- OTT/쇼핑/SNS/동영상 앱 사용 → 디지털형 근거 금지
- 휴대폰 브랜드/모델 → 근거 금지
- AI 스피커 보유 1회 → 근거 금지

────────────────────────────────────────
매우 중요한 우선순위 규칙 (디지털형 오분류 방지)
────────────────────────────────────────
1) 아래 유형들 중 강화 신호가 **2회 이상 반복되는 유형이 있다면**  
   → 해당 유형이 디지털형보다 우선입니다.
   - 실용·효율형  
   - 감정·힐링형  
   - 건강·체력관리형  
   - 여가·경험형  
   - 환경·미니멀형  
   - 가족·관계형  
   - 뷰티·자기관리형  

2) 디지털형은 "AI 관련 신호 이외의 강화 신호가 거의 없을 때"에만 선택하십시오.
   즉 아래 조건을 모두 충족할 때만 디지털형 후보가 됩니다:

   (조건 A) AI 관련 신호가 최소 2개 이상 반복  
   (조건 B) 다른 모든 유형(1,3,4,5,6,7,8)의 강화 신호가  
            각각 1회 이하이거나 단편적일 것  
   (조건 C) 감정·힐링, 실용성, 건강 관리, 여가·경험 등  
            다른 라이프스타일의 방향성이 명확하지 않을 것

3) 아래와 같은 경우 디지털형을 반드시 배제하십시오:
   - 다른 유형의 강화 신호가 더 강하고 반복적일 때  
   - AI 신호는 반복되지만, 전체 행동 맥락이 AI 중심이 아닐 때  
   - AI 신호는 있지만 소비·여가·감정·건강 등 다른 패턴이 더 지배적일 때  

────────────────────────────────────────
디지털형 primary 조건 (최종 버전)
────────────────────────────────────────
아래 3가지 모두 충족할 때만 primary로 선택하십시오:

1) AI 신호가 "주로 사용/자주 사용/능동적 활용" 형태로 2회 이상 반복  
2) 다른 7개 라이프스타일 유형 중 강화 신호가 2개 이상 반복되는 유형이 없음  
3) feature_sentence의 전반적 방향이  
   "기술 활용을 통한 문제 해결·편의·학습"에 집중되어 있을 것

────────────────────────────────────────
디지털형 secondary 조건
────────────────────────────────────────
- 다른 유형이 primary로 명확하지만  
  AI 사용 패턴도 꾸준히 드러날 때만 secondary로 포함 가능합니다.
- 단발성 AI 경험은 secondary에도 올리면 안 됩니다.



────────────────────────────────────────
[5] 건강·체력관리형(3번)
────────────────────────────────────────
강한 신호:
- 규칙적 운동(헬스, 걷기, 달리기, 홈트, 수영 등) 반복  
- 건강·체력 관리 가치관 반복  
- 건강한 식습관·다이어트 실천 반복  
- 행복 조건에서 "건강한 몸과 마음" 우선 선택  

약화/금지 신호:
- 다양한 술 종류 반복 → 감정·소비 기반  
- 스트레스 해소 음주/흡연  
- 기분 따라 음주/흡연  
- 운동 경험 단발 1회는 근거 금지  

건강형 primary 조건:
- 규칙적 운동 + 건강 가치 신호 2회 이상  
- 음주/흡연 반복 2회 이상이면 primary 금지  


────────────────────────────────────────
[6] 감정·힐링 중심형(4번) — QPOLL 기반 강화
────────────────────────────────────────
강한 신호:
- 스트레스 요인에 대한 정서적 민감성 반복  
- 스트레스 해소 방식이 '수면/휴식/명상/스파'**가 2회 이상 반복**  
- 감정 안정·힐링을 위한 명확한 목적  
- 휴가 스트레스 + 휴식 욕구 반복  
- 풍경/여행 사진이 "힐링 목적"일 때 강화  
- 정서 회복 행동(휴식·수면·명상·스파)이 2회 이상 등장  

약화/금지 신호:
- 맛있는 음식/야식/간식/배달 음식 → 감정형 근거 금지  
- 음식 기반 스트레스 해소 → 소비 행동  
- 사진 저장(셀카, 음식, 스크린샷) → 감정 신호 아님  
- 스트레스 요인만 있고 회복 행동이 없으면 근거 금지  
- 단발성 힐링 경험 1회 → 근거 금지  

감정형 primary 조건:
- 정서 민감성 + 정서 회복 행동 2회 이상  
- 음식/소비 기반 신호는 절대 활용 금지  


────────────────────────────────────────
[7] 뷰티·자기관리형(5번)
────────────────────────────────────────
강한 신호:
- 피부 상태에 대한 관심 반복  
- 스킨케어 소비 수준이 명확함  
- 성분·효과·가격·브랜드 기준으로 선택  
- 여름철 땀/피부 트러블/메이크업 무너짐 등 피부 고민 반복  

약화/금지 신호:
- 스킨케어 구매 1회 경험 → 근거 금지  
- 계절 간식, OTT, 앱 사용 → 근거 아님  

뷰티형 primary 조건:
- 피부/스킨케어 관련 신호 2회 이상  
- 성분/효과/브랜드 기준이 명확해야 강화  


────────────────────────────────────────
[8] 환경·정리·미니멀형(6번)
────────────────────────────────────────
강한 신호:
- 장바구니/에코백 상시 사용  
- 일회용품 줄이기 실천 반복  
- 재활용/업사이클링/기부 반복  
- "미니멀리스트" 자기 인식  
- 버리기 아까운 물건 → 재활용/중고 판매 반복  
- 정리/정돈 습관 지속  

약화/금지 신호:
- 기부/재활용 1회만 → 근거 금지  
- 단순 정리 기준이 실용 목적이면 실용형으로 이동  
- 미니멀 가치관이 없으면 배제  

환경·미니멀 primary 조건:
- 환경/정리 행동 2회 이상 + 미니멀 가치관  


────────────────────────────────────────
[9] 여가·경험 중심형(7번)
────────────────────────────────────────
강한 신호:
- 여행 선호 반복(유럽/동남아/일본 등)  
- 겨울 스포츠, 계절 활동, 눈썰매 등 경험  
- 콘서트·전시 등 문화 여가 경험 반복  
- 취미 기반 지출/활동 반복  
- 풍경/여행 사진 반복  
- 여름 물놀이 장소(계곡/해변/워터파크) 선호 반복  

약화/금지 신호:
- 단순 1회 여행/경험은 근거 금지  
- 음식 소비만으로는 여가형 아님  
- OTT/앱 사용은 여가형 근거 아님  

여가형 primary 조건:
- "경험 기반 즐거움/만족"이 최소 2회 이상  


────────────────────────────────────────
[10] 가족·관계 중심형(8번)
────────────────────────────────────────
강한 신호:
- 가족·친구·반려동물 중심 경험 반복  
- 관계적 만족 강조  
- 행복 조건에서 "가족·친밀감" 반복 선택  
- 중요한 기억이나 감정이 가족/반려 중심  

약화/금지 신호:
- 단순 가족 언급 1회  
- 반려동물 보유 1회  
- 여가/감정형과 혼동 금지  

가족형 primary 조건:
- 가족·관계 중심 경험이 2회 이상 반복  
- 감정형/여가형 신호와 명확히 구분될 것  


────────────────────────────────────────
[11] 판단 절차 (LLM internal)
────────────────────────────────────────
1) 모든 feature 전체 의미 구조 파악  
2) 반복되는 가치관/행동 패턴 추출  
3) 가장 일관된 primary 후보 선정  
4) 필요 시 secondary 검토  
5) 실용형 금지 규칙  
6) 건강형·디지털형 오분류 방지  
7) 최종 JSON 생성  

────────────────────────────────────────
[11-1] 라이프스타일 분류 불가 조건
────────────────────────────────────────
아래 조건 중 2개 이상이면 **어떤 라이프스타일도 선택하지 말고** 아래 형식만 출력:

{{
  "lifestyle": [],
  "message": "라이프스타일을 분류할 수 없습니다.",
  "evidence_topics": []
}}

조건:
1) 가치관·정서·여가·환경·건강·디지털 등 라이프스타일 신호가 매우 약함  
2) 8개 퍼소나 중 강화 신호가 반복되는 것이 없음  
3) 정보 대부분이 단편적(전자제품 보유, 자동차, 흡연 종류, 음주 종류 등)  
4) 라이프스타일 기준을 만족하는 문장이 1개 이하  

────────────────────────────────────────
[11-2] secondary(보조 라이프스타일) 선택 조건 — 매우 중요한 규칙
────────────────────────────────────────
secondary 라이프스타일은 아래 조건을 모두 충족할 때만 선택할 수 있습니다.

(조건 A) primary 라이프스타일의 강화 신호와는 명확히 다른 유형의 강화 신호가  
        최소 2개 이상 반복적으로 나타날 것

(조건 B) secondary 라이프스타일의 강화 신호가 "약한 신호·단편적 신호"가 아니라  
        규칙적인 행동·명확한 선호·가치관 기반일 것

(조건 C) secondary를 선택하지 않을 경우,  
        feature_sentence 전체의 의미 구조를 설명하는 데 공백이 생기는 경우에만 선택

위 조건(A, B, C) 중 하나라도 충족하지 못하면 secondary를 **절대 포함하지 마세요.**

특히 아래의 경우 secondary 선택 금지:
- 강화 신호가 1개만 있는 경우  
- 약한 신호가 여러 개 섞여 있는 경우  
- primary가 충분히 설명력을 가지는 경우  
- "애매해서 둘 다 넣는" 선택은 금지


────────────────────────────────────────
[12] 최종 출력 형식 (JSON ONLY)
────────────────────────────────────────

### ✔ Case 1 — feature_sentence ≤ 1
{{
  "lifestyle": [],
  "message": "라이프스타일을 분류할 수 없습니다.",
  "evidence_topics": []
}}

### ✔ Case 2 — 정상 분류 (1~2개)
{{
  "lifestyle": [
    {{
      "id": "1~8 중 하나",
      "lifestyle_name": "행동 유형 이름",
      "role": "primary",
      "reason": "이 유형을 선택한 핵심 근거를 2문장 이내로 간결하게 작성하세요"
    }},
    {{
      "id": "1~8 중 하나",
      "lifestyle_name": "행동 유형 이름",
      "role": "secondary",
      "reason": "보조 유형을 선택한 근거를 2문장 이내로 간결하게 작성하세요."
    }}
  ],
  "evidence_topics": ["근거가 된 topic 이름들"]
}}

출력 규칙:
- primary만 확실하면 secondary 제거  
- secondary는 정말 필요할 때만 포함  
- 이유는 모두 **2문장 이내로 간결하게 요약**  
- feature 원문 복붙 금지, **의미 요약만 사용**  
- 반드시 '이 사람이 보인 특징 → 유형의 핵심 특성과 연결' 구조로 reason 작성  
- 유형이라는 표현만 사용하고, 퍼소나라는 표현은 사용하지 마세요

이제 위 모든 규칙을 엄격히 따르고,
오직 하나의 JSON 객체만 출력하세요.
"""

