# الإجابة على الأسئلة البصرية

[[open-in-colab]]

الإجابة على الأسئلة البصرية (VQA) هي مهمة الإجابة على الأسئلة المفتوحة بناءً على صورة. عادةً ما يكون الإدخال إلى النماذج التي تدعم هذه المهمة عبارة عن مزيج من الصورة والسؤال، والمخرج هو إجابة معبر عنها باللغة الطبيعية.

فيما يلي بعض أمثلة حالات الاستخدام الجديرة بالملاحظة لـ VQA:

- تطبيقات الوصول لمساعدة الأفراد ذوي الإعاقة البصرية.
- التعليم: طرح أسئلة حول المواد المرئية المقدمة في المحاضرات أو الكتب المدرسية. يمكن أيضًا استخدام VQA في المعارض التفاعلية في المتاحف أو المواقع التاريخية.
- خدمة العملاء والتجارة الإلكترونية: يمكن لـ VQA تعزيز تجربة المستخدم من خلال السماح للمستخدمين بطرح أسئلة حول المنتجات.
- استرجاع الصور: يمكن استخدام نماذج VQA لاسترداد الصور ذات الخصائص المحددة. على سبيل المثال، يمكن للمستخدم أن يسأل "هل هناك كلب؟" للعثور على جميع الصور التي تحتوي على كلاب من مجموعة من الصور.

في هذا الدليل، ستتعلم كيفية:

- ضبط نموذج تصنيف VQA، وتحديدًا [ViLT]، على مجموعة بيانات [`Graphcore/vqa`]
- استخدام ViLT المضبوط مسبقًا للاستنتاج.
- تشغيل الاستدلال VQA بدون بيانات باستخدام نموذج توليدي، مثل BLIP-2.

## ضبط ViLT

يضم نموذج ViLT تضمين النص في محول الرؤية (ViT)، مما يسمح له بتصميم الحد الأدنى للتعلم المسبق للرؤية واللغة (VLP). يمكن استخدام هذا النموذج لعدة مهام أسفل النهر. لمهمة VQA، يتم وضع رأس مصنف أعلى (طبقة خطية أعلى الحالة المخفية النهائية للرمز `[CLS]`) ويتم تهيئتها بشكل عشوائي. وبالتالي، يتم التعامل مع الإجابة على الأسئلة البصرية كمشكلة **تصنيف**.

تتعامل النماذج الأحدث، مثل BLIP وBLIP-2 وInstructBLIP، مع VQA كمهمة توليدية. في وقت لاحق من هذا الدليل، نوضح كيفية استخدامها للاستدلال VQA بدون بيانات.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية.

```bash
pip install -q transformers datasets
```

نحن نشجعك على مشاركة نموذجك مع المجتمع. قم بتسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميله إلى 🤗 Hub.
عند المطالبة، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

دعنا نحدد نقطة تفتيش النموذج كمتغير عالمي.

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## تحميل البيانات

لأغراض التوضيح، في هذا الدليل، نستخدم عينة صغيرة جدًا من مجموعة بيانات الإجابة على الأسئلة البصرية المُعلَّمة `Graphcore/vqa`.
يمكنك العثور على مجموعة البيانات الكاملة على [🤗 Hub].

كبديل لمجموعة بيانات [`Graphcore/vqa`]]، يمكنك تنزيل نفس البيانات يدويًا من صفحة مجموعة بيانات VQA الرسمية. إذا كنت تفضل اتباع البرنامج التعليمي باستخدام بياناتك المخصصة، فراجع كيفية [إنشاء مجموعة بيانات الصور] في وثائق 🤗 Datasets.

دعنا نحمل أول 200 مثال من الانقسام التحقق من الصحة واستكشاف ميزات مجموعة البيانات:

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```

دعنا نلقي نظرة على مثال لفهم ميزات مجموعة البيانات:

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

الميزات ذات الصلة بالمهمة تشمل:

- `question`: السؤال الذي يجب الإجابة عليه من الصورة
- `image_id`: مسار الصورة التي يشير إليها السؤال
- `label`: التسميات التوضيحية

يمكننا إزالة بقية الميزات حيث لن تكون ضرورية:

```py
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

كما ترون، تحتوي ميزة `label` على عدة إجابات لنفس السؤال (تسمى `ids` هنا) التي جمعها معلّمون بشريون مختلفون.
هذا لأن إجابة السؤال يمكن أن تكون ذاتية. في هذه الحالة، السؤال هو "أين ينظر؟". قام بعض الأشخاص
وضع علامة على هذا بـ "down"، والبعض الآخر بـ "at table"، وآخر بـ "skateboard"، إلخ.

الق نظرة على الصورة واعتبر الإجابة التي ستقدمها:

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>

بسبب غموض الأسئلة والإجابات، تتم معاملة مجموعات البيانات مثل هذه كمشكلة تصنيف متعددة التصنيفات (حيث
قد تكون إجابات متعددة صالحة). علاوة على ذلك، بدلاً من إنشاء ترميز ثنائي، يتم إنشاء ترميز ناعم، بناءً على عدد المرات
تمت الإشارة إلى إجابة معينة في التسميات التوضيحية.

على سبيل المثال، في المثال أعلاه، لأن الإجابة "down" تم تحديدها بشكل أكبر بكثير من الإجابات الأخرى، فهي تحتوي على
درجة (تسمى `weight` في مجموعة البيانات) من 1.0، وبقية الإجابات لها درجات <1.0.

للاحقاً إنشاء رأس تصنيف مناسب للنموذج، دعنا ننشئ قاموسين: واحد يقوم بتعيين
اسم التسمية إلى رقم صحيح والعكس صحيح:

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()}
```

الآن بعد أن حصلنا على الخرائط، يمكننا استبدال الإجابات النصية بمعرفاتها، وتسطيح مجموعة البيانات لمزيد من المعالجة المسبقة.

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## معالجة البيانات مسبقًا

الخطوة التالية هي تحميل معالج ViLT لتحضير بيانات الصورة والنص للنموذج.
[`ViltProcessor`] يجمع بين معالج BERT للرموز ومعالج صور ViLT في معالج واحد مناسب:

```py
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

لمعالجة البيانات، نحتاج إلى تشفير الصور والأسئلة باستخدام [`ViltProcessor`]. سيستخدم المعالج
[`BertTokenizerFast`] لتوكينيز النص وإنشاء `input_ids` و`attention_mask` و`token_type_ids` لبيانات النص.
أما بالنسبة للصور، فسيستفيد المعالج من [`ViltImageProcessor`] لتصغير حجم الصورة وتطبيعها، وإنشاء `pixel_values` و`pixel_mask`.

يتم تنفيذ جميع خطوات المعالجة المسبقة هذه تلقائيًا، ولا نحتاج إلا إلى استدعاء `processor`. ومع ذلك، ما زلنا بحاجة إلى
إعداد التسميات التوضيحية المستهدفة. في هذا التمثيل، يتوافق كل عنصر مع إجابة محتملة (تسمية). للإجابات الصحيحة، يحتوي العنصر على
درجاتهم (أوزانهم)، بينما يتم تعيين العناصر المتبقية إلى الصفر.

تقوم الدالة التالية بتطبيق `processor` على الصور والأسئلة وتنسيق التسميات التوضيحية كما هو موضح أعلاه:

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()

...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score

...         targets.append(target)

...     encoding["labels"] = targets

...     return encoding
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.map`] من 🤗 Datasets. يمكنك تسريع `map` عن طريق
تعيين `batched=True` لمعالجة عدة عناصر من مجموعة البيانات في وقت واحد. في هذه المرحلة، لا تتردد في إزالة الأعمدة التي لا تحتاجها.

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

كخطوة أخيرة، قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## تدريب النموذج

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل ViLT باستخدام [`ViltForQuestionAnswering`]. حدد عدد التصنيفات
جنبا إلى جنب مع الخرائط التصنيف:

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد معلمات التدريب الخاصة بك في [`TrainingArguments`]:

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_Multiplier = 4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمعالج ومعادل البيانات.

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     tokenizer=processor,
... )
```

3. استدعاء [`~Trainer.train`] لضبط نموذجك بشكل دقيق.

```py
>>> trainer.train()
```

3. استدعاء [`~Trainer.train`] لضبط نموذجك بشكل دقيق.

```py
>>> trainer.train()
```

بمجرد الانتهاء من التدريب، شارك نموذجك على Hub باستخدام طريقة [`~Trainer.push_to_hub`] لمشاركة نموذجك النهائي على 🤗 Hub:

```py
>>> trainer.push_to_hub()
```

## الاستدلال

الآن بعد أن قمت بضبط نموذج ViLT وتحميله إلى 🤗 Hub، يمكنك استخدامه للاستدلال. أسهل
طريقة لتجربة نموذجك المضبوط مسبقًا للاستدلال هي استخدامه في [`Pipeline`].

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")
```

تم تدريب النموذج في هذا الدليل على 200 مثال فقط، لذا لا تتوقع الكثير منه. دعنا نرى إذا كان على الأقل
تعلم شيئًا من البيانات وخذ المثال الأول من مجموعة البيانات لتوضيح الاستدلال:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image, question, top_k=1)
"Where is he looking?"
[{'score': 0.5498199462890625, 'answer': 'down'}]
```

على الرغم من عدم ثقته، إلا أن النموذج قد تعلم بالفعل شيئًا ما. مع المزيد من الأمثلة وفترة التدريب الأطول، ستحصل على نتائج أفضل بكثير!

يمكنك أيضًا إعادة إنتاج نتائج الأنبوب يدويًا إذا أردت:

1. خذ صورة وسؤال، وقم بإعدادهما للنموذج باستخدام المعالج من نموذجك.
2. قم بتنفيذ الإخراج أو المعالجة المسبقة من خلال النموذج.
3. من logits، احصل على معرف الإجابة الأكثر احتمالًا، واعثر على الإجابة الفعلية في `id2label`.

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## VQA بدون تدريب

عامل النموذج السابق VQA كمهمة تصنيف. بعض النماذج الحديثة، مثل BLIP، وBLIP-2، وInstructBLIP تعالج VQA كمهمة توليدية. دعونا نأخذ [BLIP-2](../model_doc/blip-2) كمثال. لقد قدم نموذجًا جديدًا لنمذجة اللغة المرئية حيث يمكن استخدام أي مزيج من مشفر الرؤية LLM ومشفر اللغة المسبق التدريب (تعرف على المزيد في [منشور المدونة BLIP-2](https://huggingface.co/blog/blip-2)).

هذا يمكّن من تحقيق نتائج رائدة على مستوى المهام المرئية اللغوية المتعددة بما في ذلك الإجابة على الأسئلة البصرية.

دعونا نوضح كيف يمكنك استخدام هذا النموذج لـ VQA. أولاً، دعونا نحمل النموذج. هنا سنرسل النموذج بشكل صريح إلى وحدة معالجة الرسومات، إذا كانت متوفرة، والتي لم نكن بحاجة إلى القيام بها سابقًا أثناء التدريب، حيث يتعامل [`Trainer`] مع هذا تلقائيًا:

```py
>>> from transformers import AutoProcessor، Blip2ForConditionalGeneration
>>> import torch

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b"، torch_dtype=torch.float16)
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
```

يأخذ النموذج الصورة والنص كإدخال، لذا دعنا نستخدم نفس الصورة/زوج السؤال بالضبط من المثال الأول في مجموعة بيانات VQA:

```py 
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

لاستخدام BLIP-2 لمهمة الإجابة على الأسئلة المرئية، يجب أن يتبع النص الفوري تنسيقًا محددًا: `Question: {} Answer:`.

```py
>>> prompt = f"Question: {question} Answer:" 
```

الآن نحتاج إلى معالجة الصورة/الفورية باستخدام معالج النموذج، وتمرير الإدخال المعالج عبر النموذج، وفك تشفير الإخراج:

```py
>>> inputs = processor(image، text=prompt، return_tensors="pt").to(device، torch.float16)

>>> generated_ids = model.generate(**inputs، max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids، skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"إنه ينظر إلى الحشد" 
```

كما ترون، تعرف النموذج على الحشد، واتجاه الوجه (ينظر إلى الأسفل)، ومع ذلك، يبدو أنه يغفل عن حقيقة أن الحشد خلف المتزلج. ومع ذلك، في الحالات التي يكون من غير العملي فيها الحصول على مجموعات بيانات بشرية موسومة، يمكن أن ينتج هذا النهج نتائج مفيدة بسرعة.