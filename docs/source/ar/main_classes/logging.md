# التسجيل 

يحتوي 🤗 Transformers على نظام تسجيل مركزي، بحيث يمكنك بسهولة ضبط مستوى تفاصيل المكتبة. 

حاليًا، يكون مستوى التفاصيل الافتراضي للمكتبة هو `WARNING`. 

لتغيير مستوى التفاصيل، استخدم ببساطة إحدى دوال الضبط المباشرة. على سبيل المثال، إليك كيفية تغيير مستوى التفاصيل إلى مستوى `INFO`. 

```python
import transformers

transformers.logging.set_verbosity_info()
``` 

يمكنك أيضًا استخدام متغير البيئة `TRANSFORMERS_VERBOSITY` لتجاوز مستوى التفاصيل الافتراضي. يمكنك تعيينه إلى إحدى القيم التالية: `debug`، `info`، `warning`، `error`، `critical`. على سبيل المثال: 

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
``` 

بالإضافة إلى ذلك، يمكن تعطيل بعض `التحذيرات` عن طريق تعيين متغير البيئة `TRANSFORMERS_NO_ADVISORY_WARNINGS` إلى قيمة صحيحة، مثل *1*. سيؤدي هذا إلى تعطيل أي تحذير يتم تسجيله باستخدام [`logger.warning_advice`]. على سبيل المثال: 

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
``` 

فيما يلي مثال على كيفية استخدام نفس مسجل البيانات مثل المكتبة في الوحدة النمطية أو البرنامج النصي الخاص بك: 

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
``` 

جميع طرق وحدة التسجيل هذه موثقة أدناه، والأساليب الرئيسية هي 

[`logging.get_verbosity`] للحصول على مستوى التفاصيل الحالي في المسجل و 

[`logging.set_verbosity`] لضبط مستوى التفاصيل إلى المستوى الذي تختاره. وبترتيب (من الأقل إلى الأكثر تفصيلاً)، هذه المستويات (مع قيمها الصحيحة المقابلة بين قوسين) هي: 

- `transformers.logging.CRITICAL` أو `transformers.logging.FATAL` (القيمة الصحيحة، 50): قم بالإبلاغ عن الأخطاء الأكثر حرجًا فقط. 

- `transformers.logging.ERROR` (القيمة الصحيحة، 40): قم بالإبلاغ عن الأخطاء فقط. 

- `transformers.logging.WARNING` أو `transformers.logging.WARN` (القيمة الصحيحة، 30): قم بالإبلاغ عن الأخطاء والتحذيرات فقط. هذا هو مستوى التفاصيل الافتراضي الذي تستخدمه المكتبة. 

- `transformers.logging.INFO` (القيمة الصحيحة، 20): الإبلاغ عن الأخطاء والتحذيرات والمعلومات الأساسية. 

- `transformers.logging.DEBUG` (القيمة الصحيحة، 10): الإبلاغ عن جميع المعلومات. 

بشكل افتراضي، سيتم عرض مؤشرات تقدم `tqdm` أثناء تنزيل النموذج. يمكن استخدام [`logging.disable_progress_bar`] و [`logging.enable_progress_bar`] لقمع أو إلغاء قمع هذا السلوك. 

## `logging` مقابل `warnings` 

لدى Python نظامي تسجيل يتم استخدامهما غالبًا بالاقتران: `logging`، والذي تم شرحه أعلاه، و`warnings`، والذي يسمح بتصنيف إضافي للتحذيرات في دلوات محددة، على سبيل المثال، `FutureWarning` لميزة أو مسار تم إهماله بالفعل و`DeprecationWarning` للإشارة إلى الإهمال القادم. 

نحن نستخدم كلا النظامين في مكتبة `transformers`. نحن نستفيد من طريقة `captureWarning` في `logging` ونقوم بتكييفها للسماح بإدارة رسائل التحذير هذه بواسطة دوال ضبط مستوى التفاصيل أعلاه. 

ماذا يعني ذلك لمطوري المكتبة؟ يجب أن نلتزم بالمعيار التالي: 

- يجب تفضيل `التحذيرات` لمطوري المكتبة والمكتبات التي تعتمد على `transformers` 

- يجب استخدام `logging` لمستخدمي المكتبة النهائيين الذين يستخدمونها في المشاريع اليومية 

راجع مرجع طريقة `captureWarnings` أدناه. 

[[autodoc]] logging.captureWarnings 

## دوال الضبط الأساسية 

[[autodoc]] logging.set_verbosity_error 

[[autodoc]] logging.set_verbosity_warning 

[[autodoc]] logging.set_verbosity_info 

[[autodoc]] logging.set_verbosity_debug 

## الوظائف الأخرى 

[[autodoc]] logging.get_verbosity 

[[autodoc]] logging.set_verbosity 

[[autodoc]] logging.get_logger 

[[autodoc]] logging.enable_default_handler 

[[autodoc]] logging.disable_default_handler 

[[autodoc]] logging.enable_explicit_format 

[[autodoc]] logging.reset_format 

[[autodoc]] logging.enable_progress_bar 

[[autodoc]] logging.disable_progress_bar