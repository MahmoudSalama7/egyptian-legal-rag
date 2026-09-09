from __future__ import annotations

import re
from typing import Tuple


class LegalGuardrails:
    # الكلمات المفتاحية الدالة على محاولات حقن التوجيهات
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"تجاهل\s+التعليمات\s+السابقة",
        r"system\s*prompt",
        r"you\s+are\s+now",
        r"تصرف\s+كأنك",
        r"delete\s+all",
    ]

    # الكلمات التي تدل على أن السؤال قانوني
    LEGAL_KEYWORDS = [
        "عقد",
        "قانون",
        "مادة",
        "التزام",
        "حق",
        "مدني",
        "تعويض",
        "مسئولية",
        "بيع",
        "إيجار",
        "ملكية",
        "شريعة",
        "قضاء",
        "محكمة",
        "article",
        "contract",
        "law",
        "civil",
        "obligation",
        "liability",
        "damages",
    ]

    @classmethod
    def validate_input(cls, query: str) -> Tuple[bool, str]:
        """فحص السؤال قبل معالجته في محرك البحث."""
        clean_query = query.strip()

        if len(clean_query) < 3:
            return False, "السؤال قصير للغاية ولا يحتوي على استفسار قانوني مفيد."

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, clean_query, re.IGNORECASE):
                return (
                    False,
                    "تم رفض الطلب: تم رصد نمط استعلام غير مصرح به (Prompt Injection).",
                )

        return True, "Valid"

    @classmethod
    def is_legal_domain(cls, query: str) -> bool:
        """فحص ما إذا كان السؤال يقع ضمن النطاق القانوني التقريبي."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in cls.LEGAL_KEYWORDS)

    @classmethod
    def validate_output(cls, answer: str, sources: list[dict]) -> Tuple[bool, str]:
        """فحص الإجابة قبل إعادتها للعميل لضمان عدم وجود هلوسة أو خلوها من المصادر."""
        if not sources:
            return False, "تنبيه: لم يتم العثور على مواد قانونية تسند هذه الإجابة."
        return True, answer
