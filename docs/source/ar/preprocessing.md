# معالجة مسبقة

[[open-in-colab]]

قبل أن تتمكن من تدريب نموذج على مجموعة بيانات، يجب معالجتها مسبقًا إلى تنسيق إدخال النموذج المتوقع. سواء كانت بياناتك نصية أو صورًا أو صوتًا، فيجب تحويلها وتجميعها في دفعات من المنسوجات. يوفر 🤗 Transformers مجموعة من فئات المعالجة المسبقة للمساعدة في إعداد بياناتك للنموذج. في هذا البرنامج التعليمي، ستتعلم أنه بالنسبة لـ:

* للنص، استخدم [مُعرِّف الرموز](./main_classes/tokenizer) لتحويل النص إلى تسلسل من الرموز، وإنشاء تمثيل رقمي للرموز، وتجميعها في منسوجات.
* للكلام والصوت، استخدم [مستخرج الميزات](./main_classes/feature_extractor) لاستخراج ميزات متسلسلة من أشكال موجات الصوت وتحويلها إلى منسوجات.
* تستخدم مدخلات الصورة [ImageProcessor](./main_classes/image_processor) لتحويل الصور إلى منسوجات.
* تستخدم الإدخالات متعددة الوسائط [معالجًا](./main_classes/processors) لدمج مُعرِّف الرموز ومستخرج الميزات أو معالج الصور.

<Tip>

`AutoProcessor` **يعمل دائمًا** ويختار تلقائيًا الفئة الصحيحة للنموذج الذي تستخدمه، سواء كنت تستخدم مُعرِّف رموز أو معالج صور أو مستخرج ميزات أو معالجًا.

</Tip>

قبل البدء، قم بتثبيت 🤗 Datasets حتى تتمكن من تحميل بعض مجموعات البيانات لتجربتها:

```bash
pip install datasets
```

## معالجة اللغة الطبيعية

<Youtube id="Yffk5aydLzg"/>

أداة المعالجة المسبقة الرئيسية للبيانات النصية هي [مُعرِّف الرموز](main_classes/tokenizer). يقوم مُعرِّف الرموز بتقسيم النص إلى *رموز* وفقًا لمجموعة من القواعد. يتم تحويل الرموز إلى أرقام ثم إلى منسوجات، والتي تصبح إدخالات النموذج. تتم إضافة أي إدخالات إضافية مطلوبة بواسطة النموذج بواسطة مُعرِّف الرموز.

<Tip>

إذا كنت تخطط لاستخدام نموذج مُدرب مسبقًا، فمن المهم استخدام مُعرِّف الرموز المُدرب مسبقًا المقترن به. يضمن ذلك تقسيم النص بنفس الطريقة التي تم بها تقسيم مجموعة بيانات ما قبل التدريب، واستخدام نفس الرموز المقابلة للفهرس (يُشار إليها عادةً باسم *vocab*) أثناء التدريب المسبق.

</Tip>

ابدأ بتحميل مُعرِّف رموز مُدرب مسبقًا باستخدام طريقة [`AutoTokenizer.from_pretrained`]. يقوم هذا بتنزيل *vocab* الذي تم تدريب النموذج عليه:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

ثم مرر نصك إلى مُعرِّف الرموز:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

يعيد مُعرِّف الرموز قاموسًا يحتوي على ثلاثة عناصر مهمة:

* [input_ids](glossary#input-ids) هي الفهارس المقابلة لكل رمز في الجملة.
* [attention_mask](glossary#attention-mask) يشير إلى ما إذا كان يجب الاهتمام بالرمز أم لا.
* [token_type_ids](glossary#token-type-ids) يحدد التسلسل الذي ينتمي إليه الرمز عندما يكون هناك أكثر من تسلسل واحد.

أعد إدخالك عن طريق فك ترميز `input_ids`:
أعد إدخالك عن طريق فك ترميز `input_ids`:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

كما ترى، أضاف مُعرِّف الرموز رمزين خاصين - `CLS` و`SEP` (مصنف وفاصل) - إلى الجملة. لا تحتاج جميع النماذج إلى
رموز خاصة، ولكن إذا فعلوا ذلك، فإن مُعرِّف الرموز يضيفها تلقائيًا لك.

إذا كان هناك عدة جمل تريد معالجتها مسبقًا، فقم بتمريرها كقائمة إلى مُعرِّف الرموز:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### الحشو Padding

ليست الجمل دائمًا بنفس الطول، والذي يمكن أن يكون مشكلة لأن المنسوجات، إدخالات النموذج، تحتاج إلى شكل موحد. التبطين هو استراتيجية لضمان أن تكون المنسوجات مستطيلة عن طريق إضافة رمز *padding* خاص إلى الجمل الأقصر.

قم بتعيين معلمة `padding` إلى `True` لتبطين التسلسلات الأقصر في الدفعة لمطابقة التسلسل الأطول:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, Multiplier, 1, 1, 1, 1, 0، 0، 0، 0، 0، 0، 0، 0]]}
```

تم الآن تبطين الجملتين الأولى والثالثة بـ `0` لأنهما أقصر.

### البتر Truncation

من ناحية أخرى من الطيف، قد يكون التسلسل طويلًا جدًا بالنسبة للنموذج للتعامل معه. في هذه الحالة، ستحتاج إلى بتر التسلسل إلى طول أقصر.

قم بتعيين معلمة `truncation` إلى `True` لتقليم تسلسل إلى الطول الأقصى الذي يقبله النموذج:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0، 0، 0، 0، 0]]،
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0، 0، 0، 0],
                    [1, 1, 1, 1, 1, 1, 1، 1، 1، 1، 1، 1، 1، 1، 1، 1],
                    [1، 1، 1، 1، 1، 1، 1، 0، 0، 0، 0، 0، 0، 0، 0، 0]]}
```

<Tip>
<Tip>

تحقق من دليل المفاهيم [Padding and truncation](./pad_truncation) لمعرفة المزيد حول وسادات وحجج البتر المختلفة.

</Tip>

### بناء التنسرات Build tensors

أخيرًا، تريد أن يقوم مُعرِّف الرموز بإرجاع التنسرات الفعلية التي يتم تغذيتها في النموذج.

قم بتعيين معلمة `return_tensors` إلى إما `pt` لـ PyTorch، أو `tf` لـ TensorFlow:

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32
## رؤية الكمبيوتر

بالنسبة لمهام رؤية الكمبيوتر، ستحتاج إلى معالج صور [image processor](main_classes/image_processor) لإعداد مجموعة البيانات الخاصة بك للنمذجة. تتكون معالجة الصور المسبقة من عدة خطوات لتحويل الصور إلى الإدخال الذي يتوقعه النموذج. وتشمل هذه الخطوات، على سبيل المثال لا الحصر، تغيير الحجم والتطبيع وتصحيح قناة الألوان وتحويل الصور إلى تنسورات.

<Tip>

عادة ما تتبع معالجة الصور المسبقة شكلاً من أشكال زيادة الصور. كل من معالجة الصور المسبقة وزيادة الصور تحويل بيانات الصور، ولكنها تخدم أغراضًا مختلفة:

* تغيير الصور عن طريق زيادة الصور بطريقة يمكن أن تساعد في منع الإفراط في التجهيز وزيادة متانة النموذج. يمكنك أن تكون مبدعًا في كيفية زيادة بياناتك - ضبط السطوع والألوان، والمحاصيل، والدوران، تغيير الحجم، التكبير، إلخ. ومع ذلك، كن حذرًا من عدم تغيير معنى الصور باستخدام الزيادات الخاصة بك.
* تضمن معالجة الصور المسبقة أن تتطابق الصور مع تنسيق الإدخال المتوقع للنموذج. عند ضبط نموذج رؤية الكمبيوتر بدقة، يجب معالجة الصور بالضبط كما كانت عند تدريب النموذج لأول مرة.

يمكنك استخدام أي مكتبة تريدها لزيادة الصور. لمعالجة الصور المسبقة، استخدم `ImageProcessor` المرتبط بالنموذج.

</Tip>

قم بتحميل مجموعة بيانات [food101](https://huggingface.co/datasets/food101) (راجع دليل 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub) لمزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج الصور مع مجموعات بيانات رؤية الكمبيوتر:

<Tip>

استخدم معلمة `split` من 🤗 Datasets لتحميل عينة صغيرة فقط من الانقسام التدريبي نظرًا لأن مجموعة البيانات كبيرة جدًا!

</Tip>

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

بعد ذلك، الق نظرة على الصورة مع ميزة 🤗 Datasets [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=image#datasets.Image):

```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

قم بتحميل معالج الصور باستخدام [`AutoImageProcessor.from_pretrained`]:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

أولاً، دعنا نضيف بعض الزيادات إلى الصور. يمكنك استخدام أي مكتبة تفضلها، ولكن في هذا الدليل، سنستخدم وحدة [`transforms`](https://pytorch.org/vision/stable/transforms.html) من torchvision. إذا كنت مهتمًا باستخدام مكتبة زيادة بيانات أخرى، فتعرف على كيفية القيام بذلك في [دفاتر Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) أو [دفاتر Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb).

1. هنا نستخدم [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) لربط بعض التحولات معًا - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) و [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html).
لاحظ أنه بالنسبة لتغيير الحجم، يمكننا الحصول على متطلبات حجم الصورة من `image_processor`. بالنسبة لبعض النماذج، يُتوقع ارتفاع وعرض محددان بالضبط، بينما بالنسبة للنماذج الأخرى، يتم تحديد `shortest_edge` فقط.

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. يقبل النموذج [`pixel_values`](model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel.forward.pixel_values)
كإدخال له. يمكن لـ `ImageProcessor` التعامل مع تطبيع الصور، وتوليد التنسورات المناسبة.
قم بإنشاء دالة تجمع بين زيادة الصور ومعالجة الصور المسبقة لمجموعة من الصور وتوليد `pixel_values`:

```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

في المثال أعلاه، قمنا بتعيين `do_resize=False` لأننا قمنا بالفعل بتغيير حجم الصور في تحويل زيادة الصور،
واستفدنا من خاصية `size` من `image_processor` المناسب. إذا لم تقم بتغيير حجم الصور أثناء زيادة الصور،
فاترك هذا المعلمة. بشكل افتراضي، ستتعامل `ImageProcessor` مع تغيير الحجم.

إذا كنت ترغب في تطبيع الصور كجزء من تحويل زيادة الصور، فاستخدم قيم `image_processor.image_mean`،
و `image_processor.image_std`.
</Tip>

3. ثم استخدم 🤗 Datasets[`~datasets.Dataset.set_transform`] لتطبيق التحولات أثناء التنقل:
```py
>>> dataset.set_transform(transforms)
```

4. الآن عند الوصول إلى الصورة، ستلاحظ أن معالج الصور قد أضاف `pixel_values`. يمكنك تمرير مجموعة البيانات المعالجة إلى النموذج الآن!

```py
>>> dataset[0].keys()
```

هكذا تبدو الصورة بعد تطبيق التحولات. تم اقتصاص الصورة بشكل عشوائي وتختلف خصائص الألوان بها.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

بالنسبة للمهام مثل الكشف عن الأشياء، والتجزئة الدلالية، والتجزئة المثالية، والتجزئة الشاملة، يوفر `ImageProcessor`
أساليب ما بعد المعالجة. تحول هذه الأساليب المخرجات الخام للنموذج إلى تنبؤات ذات معنى مثل صناديق الحدود،
أو خرائط التجزئة.

</Tip>

### الحشو Pad

في بعض الحالات، على سبيل المثال، عند ضبط نموذج [DETR](./model_doc/detr) بدقة، يقوم النموذج بتطبيق زيادة المقياس في وقت التدريب. قد يتسبب ذلك في اختلاف أحجام الصور في دفعة واحدة. يمكنك استخدام [`DetrImageProcessor.pad`]
من [`DetrImageProcessor`] وتعريف `collate_fn` مخصص لتجميع الصور معًا.

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## متعدد الوسائط Mulimodal

بالنسبة للمهام التي تتضمن إدخالات متعددة الوسائط، ستحتاج إلى معالج [processor](main_classes/processors) لإعداد مجموعة البيانات الخاصة بك للنمذجة. يربط المعالج بين كائنين للمعالجة مثل الرموز المميزة ومستخرج الميزات.

قم بتحميل مجموعة بيانات [LJ Speech](https://huggingface.co/datasets/lj_speech) (راجع دليل 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub) لمزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج للتعرف التلقائي على الكلام (ASR):

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

بالنسبة لـ ASR، فأنت تركز بشكل أساسي على `audio` و `text` لذا يمكنك إزالة الأعمدة الأخرى:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

الآن الق نظرة على أعمدة `audio` و `text`:
```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

الآن الق نظرة على أعمدة `audio` و `text`:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

تذكر أنه يجب عليك دائمًا [إعادة أخذ العينات](preprocessing#audio) لمعدل أخذ العينات في مجموعة البيانات الصوتية الخاصة بك لمطابقة معدل أخذ العينات في مجموعة البيانات المستخدمة لتدريب النموذج مسبقًا!

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

قم بتحميل معالج باستخدام [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. قم بإنشاء دالة لمعالجة بيانات الصوت الموجودة في `array` إلى `input_values`، ورموز `text` إلى `labels`. هذه هي المدخلات للنموذج:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. قم بتطبيق دالة `prepare_dataset` على عينة:

```py
>>> prepare_dataset(lj_speech[0])
```

لقد أضاف المعالج الآن `input_values` و `labels`، وتم أيضًا إعادة أخذ العينات لمعدل أخذ العينات بشكل صحيح إلى 16 كيلو هرتز. يمكنك تمرير مجموعة البيانات المعالجة إلى النموذج الآن!