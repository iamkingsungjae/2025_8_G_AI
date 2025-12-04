"""카테고리 분류기"""
import json
from typing import Dict, Any, List, Optional
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)


class CategoryClassifier:
    """LLM으로 메타데이터를 카테고리별로 분류"""

    def __init__(self, category_config: Dict[str, Any], api_key: str):
        """
        Args:
            category_config: 카테고리 설정 딕셔너리
            api_key: Anthropic API 키
        """
        self.category_config = category_config
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"  #  haiku 사용

    def _build_prompt(self, metadata: Dict[str, Any]) -> str:
        """카테고리 설명 + 메타데이터를 포함한 LLM용 프롬프트 생성"""
        
        # 카테고리 설명
        category_desc = "\n".join([
            f"- {cat}: {info.get('description', ', '.join(info.get('keywords', [])))}"
            for cat, info in self.category_config.items()
        ])

        # 키: 값 형식으로 메타데이터 나열
        meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
        meta_text = "\n".join(meta_lines)

        # 사용 가능한 키 이름 목록
        meta_keys = ", ".join(metadata.keys())

        prompt = f"""
당신은 메타데이터를 카테고리로 분류하는 전문가입니다.

다음은 사용할 수 있는 카테고리 목록과 설명입니다:
{category_desc}

다음은 분류해야 할 메타데이터입니다 (키: 값 형식):
{meta_text}

이때 사용할 수 있는 '키 이름' 목록은 다음과 같습니다:
{meta_keys}

당신의 작업:
각 메타데이터의 "키 이름"을 정확히 하나의 카테고리에 배정하세요.

출력은 반드시 아래 JSON 형식을 따라야 합니다 (예시는 구조만 참고):

{{
  "기본정보": ["지역", "성별"],
  "미디어": ["조건"],
  "스트레스": [],
  "기타": []
}}

카테고리 작업 규칙:
1. 각 메타데이터 키는 반드시 1개의 카테고리에만 속해야 합니다.
2. "키: 값" 전체를 쓰지 말고, 오직 '키 이름'만 써야 합니다.
3. 값(value)이나 새로운 문장, 설명문, 여분의 텍스트는 절대 포함하지 마세요.
4. 반드시 위에 나열된 키 이름만 사용하세요. 값이나 문장을 JSON에 넣으면 안 됩니다.

JSON만 반환하세요:
"""
        return prompt.strip()

    def classify(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        메타데이터를 LLM을 통해 카테고리별로 분류

        Returns:
            {"카테고리명": ["키: 값", "키: 값", ...]}
        """
        if not metadata:
            return {}

        prompt = self._build_prompt(metadata)

        try:
            # LLM 호출
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0  # 30초 타임아웃
            )
            
            raw_output = response.content[0].text.strip()

            # JSON 파싱
            try:
                mapping_tokens = self._parse_llm_output(raw_output)
            except Exception as parse_err:
                logger.warning(f"[카테고리 분류] JSON 파싱 실패: {parse_err}, 원본: {raw_output[:200]}")
                raise

            # 토큰들을 실제 메타데이터 키로 매핑
            categorized: Dict[str, List[str]] = {}
            used_keys: set = set()

            for cat, tokens in mapping_tokens.items():
                for token in tokens:
                    meta_key = self._match_llm_token_to_key(token, metadata, used_keys)
                    if meta_key is None:
                        continue
                    categorized.setdefault(cat, []).append(f"{meta_key}: {metadata[meta_key]}")
                    used_keys.add(meta_key)

            # 아무 것도 매핑 안 됐으면 rule-based로 폴백
            if not categorized:
                logger.warning(f"[WARN] LLM 기반 분류 결과 매핑 실패 -> rule-based로 대체 (메타데이터: {metadata}, 매핑 토큰: {mapping_tokens})")
                return self._rule_based_classify(metadata)

            logger.info(f"[카테고리 분류] {dict(categorized)}")
            return categorized

        except Exception as e:
            logger.warning(f"[WARN] LLM 분류/파싱 실패 ({e}) -> rule-based로 대체")
            return self._rule_based_classify(metadata)

    def _parse_llm_output(self, raw_output: str) -> Dict[str, List[str]]:
        """LLM이 반환한 raw 문자열을 JSON으로 파싱"""
        # 코드블록 제거
        if "```json" in raw_output:
            try:
                raw_output = raw_output.split("```json", 1)[1].split("```", 1)[0].strip()
            except:
                pass
        elif "```" in raw_output:
            try:
                raw_output = raw_output.split("```", 1)[1].split("```", 1)[0].strip()
            except:
                pass

        # JSON 파싱
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning(f"[카테고리 분류] JSON 파싱 오류: {e}, 원본: {raw_output[:200]}")
            raise

        # 값들을 전부 리스트[str] 형태로 정규화
        mapping_tokens: Dict[str, List[str]] = {}
        for cat, vals in parsed.items():
            if isinstance(vals, list):
                tokens = [str(v).strip() for v in vals if str(v).strip()]
            elif isinstance(vals, str):
                tokens = [vals.strip()] if vals.strip() else []
            elif isinstance(vals, dict):
                tokens = [str(k).strip() for k in vals.keys() if str(k).strip()]
            else:
                tokens = [str(vals).strip()]

            if tokens:
                mapping_tokens[cat] = tokens

        return mapping_tokens

    def _match_llm_token_to_key(self, token: str, metadata: Dict[str, Any], used_keys: set) -> Optional[str]:
        """LLM이 JSON에 넣은 토큰을 실제 메타데이터 키로 매핑"""
        t = token.strip()
        if not t:
            return None

        # 1) 정확히 같은 키 이름
        if t in metadata and t not in used_keys:
            return t

        # 2) "키: 값" 형식으로 온 경우
        if ":" in t:
            left = t.split(":", 1)[0].strip()
            if left in metadata and left not in used_keys:
                return left

        # 3) 값 문자열과의 유사 매칭
        t_lower = t.lower()
        for meta_key, meta_value in metadata.items():
            if meta_key in used_keys:
                continue
            v_lower = str(meta_value).lower()

            if t_lower in v_lower or v_lower in t_lower:
                return meta_key

        return None

    def _rule_based_classify(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """백업용: 기존 키워드 기반 규칙 분류"""
        categorized: Dict[str, List[str]] = {}
        for meta_key, meta_value in metadata.items():
            matched_categories = self._match_categories(meta_value)
            for category in matched_categories:
                categorized.setdefault(category, []).append(f"{meta_key}: {meta_value}")
        return categorized

    def _match_categories(self, value) -> List[str]:
        matched: List[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    matched.extend(self._match_single_value(item))
        elif isinstance(value, str):
            matched = self._match_single_value(value)
        return list(set(matched))

    def _match_single_value(self, value: str) -> List[str]:
        matched: List[str] = []
        value_lower = value.lower()
        for category_name, category_info in self.category_config.items():
            for keyword in category_info.get("keywords", []):
                if keyword.lower() in value_lower:
                    matched.append(category_name)
                    break
        return matched


