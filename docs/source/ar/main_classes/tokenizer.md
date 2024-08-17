# معالج الرموز 

معالج الرموز هو المسؤول عن إعداد المدخلات للنموذج. تحتوي المكتبة على معالجات رموز لجميع النماذج. معظم معالجات الرموز متوفرة بنكهتين: تنفيذ Python كامل وتنفيذ "سريع" يعتمد على مكتبة Rust [🤗 Tokenizers](https://github.com/huggingface/tokenizers). تسمح عمليات التنفيذ "السريعة" بما يلي:

1. تسريع كبير، خاصة عند إجراء التمييز بين الدفعات
2. أساليب إضافية للربط بين السلسلة الأصلية (الحروف والكلمات) ومساحة الرمز (على سبيل المثال، الحصول على فهرس الرمز الذي يتضمن حرفًا معينًا أو تسلسل الأحرف المقابل لرمز معين).

تنفذ الفئات الأساسية [PreTrainedTokenizer] و [PreTrainedTokenizerFast] الأساليب الشائعة لتشفير إدخالات السلسلة في إدخالات النموذج (انظر أدناه) وإنشاء/حفظ معالجات الرموز Python و "Fast" إما من ملف أو دليل محلي أو من معالج رموز مسبق التدريب يوفره المكتبة (تم تنزيله من مستودع AWS S3 الخاص بـ HuggingFace). يعتمد كلاهما على [~ tokenization_utils_base.PreTrainedTokenizerBase] الذي يحتوي على الأساليب الشائعة، و [~ tokenization_utils_base.SpecialTokensMixin].

وبالتالي، فإن [PreTrainedTokenizer] و [PreTrainedTokenizerFast] تنفذ الأساليب الرئيسية لاستخدام جميع معالجات الرموز:

- معالجة الرموز (تقسيم السلاسل إلى سلاسل رموز فرعية)، وتحويل سلاسل الرموز إلى معرفات والعكس، والترميز/فك الترميز (أي معالجة الرموز وتحويلها إلى أعداد صحيحة).
- إضافة رموز جديدة إلى المفردات بطريقة مستقلة عن البنية الأساسية (BPE، SentencePiece...).
- إدارة الرموز الخاصة (مثل القناع، وبداية الجملة، وما إلى ذلك): إضافتها، وتعيينها إلى سمات في معالج الرموز للوصول السهل، والتأكد من عدم تقسيمها أثناء معالجة الرموز.

يحتوي [BatchEncoding] على إخراج أساليب الترميز لـ [~ tokenization_utils_base.PreTrainedTokenizerBase] (`__call__`، `encode_plus` و `batch_encode_plus`) ومشتق من قاموس Python. عندما يكون معالج الرموز عبارة عن معالج رموز Python نقي، يتصرف هذا الصنف مثل قاموس Python القياسي ويحتوي على إدخالات النموذج المختلفة التي تحسبها هذه الأساليب (`input_ids`، `attention_mask`...). عندما يكون معالج الرموز عبارة عن "معالج رموز سريع" (أي مدعوم من مكتبة HuggingFace [tokenizers](https://github.com/huggingface/tokenizers))، توفر هذه الفئة بالإضافة إلى ذلك عدة أساليب محاذاة متقدمة يمكن استخدامها للربط بين السلسلة الأصلية (الحروف والكلمات) ومساحة الرمز (على سبيل المثال، الحصول على فهرس الرمز الذي يتضمن حرفًا معينًا أو تسلسل الأحرف المقابل لرمز معين).

## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer

- __call__
- add_tokens
- add_special_tokens
- apply_chat_template
- batch_decode
- decode
- encode
- push_to_hub
- all

## PreTrainedTokenizerFast

يعتمد [PreTrainedTokenizerFast] على مكتبة [tokenizers](https://huggingface.co/docs/tokenizers). يمكن تحميل معالجات الرموز التي تم الحصول عليها من مكتبة 🤗 tokenizers إلى 🤗 transformers ببساطة شديدة. راجع صفحة [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) لمعرفة كيفية القيام بذلك.

[[autodoc]] PreTrainedTokenizerFast

- __call__
- add_tokens
- add_special_tokens
- apply_chat_template
- batch_decode
- decode
- encode
- push_to_hub
- all

## BatchEncoding

[[autodoc]] BatchEncoding