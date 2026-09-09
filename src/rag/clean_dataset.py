from __future__ import annotations

import json
import re
from pathlib import Path

# أنماط العناوين الهيكلية التي تأتي في ذيل المادة فقط
AR_TRAILING_PATTERNS = [
    r"(?:\n)\s*(?:القسم|الكتاب|الباب|الفصل)\s+.*$",
    r"(?:\n)\s*(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً)[:\-–\s].*$",
    r"(?:\n)\s*[0-9٠-٩]+[\s\-–]+(?:تطبيق القانون|الشخص الطبيعي|الشخص الاعتباري|الجمعيات|أركان|المسئولية|التنفيذ|أسباب|أحكام|طرق|نطاقه|وسائل|عقد|الحقوق).*$",
    r"(?:\n)\s*(?:تنازع القوانين|الإعسار|الشرط والأجل|عدم القابلية|حوالة الدين|الوفاء بمقابل|التجديد والإنابة|المقاصة|اتحاد الذمة|الإبراء|استحالة التنفيذ|التقادم المسقط|بيع الوفاء|بيع ملك الغير|بيع الحقوق المتنازع عليها|بيع التركة|البيع في مرض الموت|بيع النائب لنفسه|المقايضة|الهبة|الشركة|القرض|الدخل الدائم|الصلح|الحراسة|المقامرة|المرتب|الكفالة|حق الملكية|الشيوع الإجباري|ملكية الأسرة|ملكيات الطبقات|حق الانتفاع|حق الاستعمال وحق السكن|حق الحكر|حق الارتفاق|الرهن الرسمي|حق الاختصاص|الرهن الحيازي|حقوق الامتياز).*$",
]

EN_TRAILING_PATTERNS = [
    r"(?:\n)\s*(?:FIRST|SECOND|THIRD)?\s*(?:PART|BOOK|CHAPTER|SECTION)\s+[I|V|X|\d]+.*$",
    r"(?:\n)\s*\d+[\.\-\s]+(?:The Application|Persons|Individuals|Juristic|Elements|Contracts|Unlawful|Specific|Means|Insolvency|Conditional|Time|Alternative|Plurality|Joint|Assignment|Payment|Extinction|Loans|Compromise|Leases|Gifts|Partnership|Suretyship|The Right|Acquisition).*$",
    r"(?:\n)\s*(?:Conflicts of law|Consent|Objects?|Nullity|Consideration|Object|Specific Performance|General Provisions).*$",
]


def clean_arabic_text(text: str) -> str:
    if not text:
        return ""

    # 1. إزالة كلمة مادة من البداية
    text = re.sub(r"^\s*\(?\s*(?:مادة|ماده)\s*\(?\s*", "", text)

    # 2. إزالة الأرقام المقلوبة والبادئة بصفر في السطر الأول فقط
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        # إزالة أرقام مثل ٠١( أو ٠٠١( أو ٠٢( دون المساس بالفقرات السليمة مثل (١) أو (٢)
        first_line = re.sub(
            r"^[\(\)\s]*[0٠][0-9٠-٩]{1,4}[\(\)\s]*", "", first_line
        ).strip()
        lines[0] = first_line
        text = "\n".join(line for line in lines if line and line != "ا")

    # 3. إزالة ترويسات الأبواب والفصول من الذيول
    for pat in AR_TRAILING_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.IGNORECASE)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and line.strip() != "ا"
    ]
    return "\n".join(lines).strip()


def clean_english_text(text: str) -> str:
    if not text:
        return ""
    for pat in EN_TRAILING_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def run_cleanup(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    art_dict = {a["article_number"]: a for a in articles}

    # 1. فصل المادة 1022 المستقلة عن 1021 لضبط العدد (1148)
    if 1022 not in art_dict:
        art_dict[1022] = {
            "article_number": 1022,
            "book": None,
            "chapter": None,
            "section": None,
            "topic": None,
            "text_ar": "نفقات الأعمال اللازمة لاستعمال حق الارتفاق والمحافظة عليه تكون على عاتق مالك العقار المرتفق ما لم يشترط غير ذلك.\nفإذا كان مالك العقار المرتفق به هو المكلف بأن يقوم بتلك الأعمال على نفقته، كان له دائماً أن يتخلص من هذا التكليف بالتخلي عن العقار المرتفق به كله أو بعضه لمالك العقار المرتفق.\nوإذا كانت الأعمال نافعة أيضاً لمالك العقار المرتفق به، كانت نفقة الصيانة على الطرفين كل بنسبة ما يعود عليه من الفائدة.",
            "text_en": "In the absence of an agreement to the contrary, the cost of the necessary works for the use and preservation of the servitude must be borne by the owner of the dominant tenement.\nIf the owner of the servient tenement is responsible for carrying out these works at his own cost, he has always the right to free himself of this burden by abandoning the servient tenement wholly or in part to the owner of the dominant property.\nIf the works also benefit the owner of the servient tenement, the cost of upkeep falls on the two parties in proportion to the profit derived by each of them.",
            "is_repealed": False,
            "source_page": 147,
            "citation": "Egyptian Civil Code, Article 1022",
        }
    if 1021 in art_dict:
        art_dict[1021]["text_ar"] = (
            "لا يلزم مالك العقار المرتفق به أن يقوم بأي عمل لمصلحة العقار المرتفق إلا أن يكون عملاً إضافياً يقتضيه استعمال الارتفاق على الوجه المألوف ما لم يشترط غير ذلك."
        )
        art_dict[1021]["text_en"] = (
            "In the absence of an agreement to the contrary, the owner of the servient tenement is under no obligation to carry out work for the benefit of the dominant tenement, unless it is an accessory work necessitated by the normal use of the servitude."
        )

    # 2. تنظيف عام لجميع المواد
    for item in art_dict.values():
        if not item.get("is_repealed"):
            item["text_ar"] = clean_arabic_text(item["text_ar"])
            item["text_en"] = clean_english_text(item["text_en"])

    # 3. تثبيت نصوص المواد الحساسة بعد التنظيف لتفادي حذف الـ regex لها
    # المادة 279
    if 279 in art_dict:
        art_dict[279]["text_ar"] = (
            "التضامن بين الدائنين أو بين المدينين لا يفترض ، وإنما يكون بناء على اتفاق أو نص في القانون."
        )
        art_dict[279]["text_en"] = (
            "Solidarity between creditors or between debtors is not presumed. It is created by agreement or by law."
        )

    # المادة 884
    if 884 in art_dict:
        art_dict[884]["text_ar"] = (
            "للمحكمة بناء على طلب أحد الورثة أو المصفى أو ذى الشأن أن تقرر عزل المصفى واستبدال غيره به إذا وجد سبب يبرر ذلك."
        )
        art_dict[884]["text_en"] = (
            "The Court may, at the request of one of the heirs, the administrator, or any interested party, discharge the administrator and replace him by another if there are reasons justifying such action."
        )

    # المادة 901
    if 901 in art_dict:
        art_dict[901]["text_ar"] = (
            "تسلم المحكمة إلى كل وارث يقدم إعلاماً شرعياً بالوراثة أو ما يقوم مقام هذا الإعلام ، شهادة تقرر حقه في الإرث وتبين ما آل إليه من أموال التركة."
        )
        art_dict[901]["text_en"] = (
            "The Court will give to each heir who produces an Elam Charei, or any other equivalent document as to the inheritance, a certificate establishing his rights in the inheritance, the extent of his share therein and the estate property devolving on him."
        )

    # المادة 935
    if 935 in art_dict:
        art_dict[935]["text_ar"] = (
            "الشفعة رخصة تجيز في بيع العقار الحلول محل المشتري في الأحوال وبالشروط المنصوص عليها في المواد التالية."
        )
        art_dict[935]["text_en"] = (
            "Preemption is the opportunity that a person has to substitute himself in a sale of immovable property in the place of the purchaser, in the cases and subject to the conditions laid down in the following articles."
        )

    # 4. حفظ الناتج مرتباً
    cleaned_records = [art_dict[k] for k in sorted(art_dict.keys()) if k <= 1149]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    print(f"Cleaned dataset saved successfully to: {output_path}")
    print(f"Total valid articles: {len(cleaned_records)}")


if __name__ == "__main__":
    run_cleanup(
        "data/processed/civil_code.json", "data/processed/civil_code_ready.json"
    )
