# تحويل النص إلى كلام

[[open-in-colab]]

تحويل النص إلى كلام (TTS) هي مهمة إنشاء كلام طبيعي من نص مكتوب، حيث يمكن توليد الكلام بعدة لغات ولعدة متحدثين. هناك العديد من نماذج تحويل النص إلى كلام المتاحة حاليًا في 🤗 Transformers، مثل [Bark](../model_doc/bark)، [MMS](../model_doc/mms)، [VITS](../model_doc/vits) و [SpeechT5](../model_doc/speecht5).

يمكنك بسهولة إنشاء صوت باستخدام خط أنابيب "text-to-audio" (أو الاسم المستعار الخاص به - "text-to-speech"). يمكن لبعض النماذج، مثل Bark، أيضًا أن تكون مشروطة لتوليد اتصالات غير لفظية مثل الضحك والتنهد والبكاء، أو حتى إضافة الموسيقى.
فيما يلي مثال على كيفية استخدام خط أنابيب "text-to-speech" مع Bark:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="suno/bark-small")
>>> text = "[clears throat] This is a test ... and I just took a long pause."
>>> output = pipe(text)
```

فيما يلي مقتطف من التعليمات البرمجية التي يمكنك استخدامها للاستماع إلى الصوت الناتج في دفتر الملاحظات:

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

للحصول على المزيد من الأمثلة على ما يمكن أن تفعله Bark ونماذج TTS الأخرى المُدربة مسبقًا، راجع [دورة الصوت](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models) الخاصة بنا.

إذا كنت ترغب في ضبط نموذج TTS، فإن النماذج الوحيدة لتحويل النص إلى كلام المتاحة حاليًا في 🤗 Transformers هي [SpeechT5](model_doc/speecht5) و [FastSpeech2Conformer](model_doc/fastspeech2_conformer)، على الرغم من أنه سيتم إضافة المزيد في المستقبل. تم تدريب SpeechT5 مسبقًا على مجموعة من بيانات تحويل الكلام إلى نص وتحويل النص إلى كلام، مما يسمح لها بتعلم مساحة موحدة من التمثيلات المخفية التي يشاركها كل من النص والكلام. وهذا يعني أنه يمكن ضبط نفس النموذج المُدرب مسبقًا لمهام مختلفة. علاوة على ذلك، يدعم SpeechT5 متحدثين متعددين من خلال تضمين المتحدث x.

يوضح باقي هذا الدليل كيفية:

1. ضبط نموذج [SpeechT5](../model_doc/speecht5) الذي تم تدريبه في الأصل على الكلام باللغة الإنجليزية على اللغة الهولندية (`nl`)، وهي مجموعة فرعية من مجموعة بيانات [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli).
2. استخدام نموذجك المحسن للاستنتاج بطريقتين: باستخدام خط أنابيب أو مباشرة.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install datasets soundfile speechbrain accelerate
```

قم بتثبيت 🤗Transformers من المصدر حيث لم يتم دمج جميع ميزات SpeechT5 في إصدار رسمي بعد:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

للحصول على هذا الدليل، ستحتاج إلى GPU. إذا كنت تعمل في دفتر ملاحظات، فقم بتشغيل السطر التالي للتحقق مما إذا كانت GPU متوفرة:

```bash
!nvidia-smi
```

أو بديل لـ GPUs AMD:

```bash
!rocm-smi
```

</Tip>

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميل ومشاركة نموذجك مع المجتمع. عندما يتم مطالبتك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة البيانات
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة البيانات

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) هي مجموعة بيانات صوتية متعددة اللغات واسعة النطاق تتكون من
بيانات مستخرجة من تسجيلات أحداث البرلمان الأوروبي من 2009 إلى 2020. يحتوي على بيانات صوتية منقحة لخمس عشرة
لغة أوروبية. في هذا الدليل، نستخدم المجموعة الفرعية للغة الهولندية، لا تتردد في اختيار مجموعة فرعية أخرى.

لاحظ أن VoxPopuli أو أي مجموعة بيانات أخرى للتعرف التلقائي على الكلام (ASR) قد لا تكون الخيار الأنسب
لتدريب نماذج TTS. الميزات التي تجعلها مفيدة لـ ASR، مثل الضوضاء الخلفية المفرطة، غير مرغوب فيها عادة في TTS. ومع ذلك، يمكن أن يكون العثور على مجموعات بيانات TTS عالية الجودة ومتعددة اللغات والمتعددة المتحدثين أمرًا صعبًا للغاية.

دعنا نحمل البيانات:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20968 مثال يجب أن يكون كافيًا للضبط الدقيق. يتوقع SpeechT5 أن يكون لمجموعة البيانات معدل أخذ عينات يبلغ 16 كيلو هرتز، لذا
تأكد من أن الأمثلة في مجموعة البيانات تلبي هذا الشرط:

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## معالجة البيانات مسبقًا

دعنا نبدأ بتحديد نقطة تفتيش النموذج لتحميلها وتحميل المعالج المناسب:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### تنظيف النص لتوكينيز SpeechT5

ابدأ بتنظيف بيانات النص. ستحتاج إلى الجزء المعالج من المعالج لمعالجة النص:

```py
>>> tokenizer = processor.tokenizer
```

تحتوي أمثلة مجموعة البيانات على ميزات `raw_text` و `normalized_text`. عند اتخاذ قرار بشأن الميزة التي سيتم استخدامها كإدخال نصي،
ضع في اعتبارك أن معالج SpeechT5 لا يحتوي على أي رموز للأرقام. في `normalized_text` يتم كتابة الأرقام
كـنص. وبالتالي، فهو مناسب بشكل أفضل، ونوصي باستخدام `normalized_text` كنص إدخال.

نظرًا لأن SpeechT5 تم تدريبه على اللغة الإنجليزية، فقد لا يتعرف على أحرف معينة في مجموعة البيانات الهولندية. إذا
تم تركها كما هي، فسيتم تحويل هذه الأحرف إلى رموز `<unk>`. ومع ذلك، في اللغة الهولندية، يتم استخدام أحرف معينة مثل `à`
للتأكيد على المقاطع. للحفاظ على معنى النص، يمكننا استبدال هذا الحرف بـ `a` عادي.

لتحديد الرموز غير المدعومة، استخرج جميع الأحرف الفريدة في مجموعة البيانات باستخدام `SpeechT5Tokenizer`
التي تعمل مع الأحرف كرموز. للقيام بذلك، اكتب وظيفة `extract_all_chars` mapping التي تقوم بدمج
نصوص من جميع الأمثلة في سلسلة واحدة وتحويلها إلى مجموعة من الأحرف.
تأكد من تعيين `batched=True` و `batch_size=-1` في `dataset.map()` بحيث تكون جميع النصوص متاحة مرة واحدة لوظيفة الخريطة.

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["normalized_text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

الآن لديك مجموعتان من الأحرف: واحدة مع المفردات من مجموعة البيانات والأخرى مع المفردات من المعالج.
لتحديد أي أحرف غير مدعومة في مجموعة البيانات، يمكنك أخذ الفرق بين هاتين المجموعتين. ستتضمن النتيجة
تعيين الأحرف الموجودة في مجموعة البيانات ولكن ليس في المعالج.

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
``````py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

لمعالجة الأحرف غير المدعومة التي تم تحديدها في الخطوة السابقة، حدد وظيفة تقوم بتعيين هذه الأحرف إلى
رموز صالحة. لاحظ أن المسافات يتم استبدالها بالفعل بـ `▁` في المعالج ولا تحتاج إلى معالجة منفصلة.

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

الآن بعد أن تعاملت مع الأحرف الخاصة في النص، حان الوقت للتركيز على بيانات الصوت.

### المتحدثون

تتضمن مجموعة بيانات VoxPopuli كلامًا من متحدثين متعددين، ولكن كم عدد المتحدثين الممثلين في مجموعة البيانات؟ لتحديد ذلك، يمكننا
عد عدد المتحدثين الفريدين وعدد الأمثلة التي يساهم بها كل متحدث في مجموعة البيانات.
مع ما مجموعه 20968 مثال في مجموعة البيانات، ستمنحنا هذه المعلومات فهمًا أفضل لتوزيع المتحدثين والأمثلة في البيانات.

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

من خلال رسم مخطط توزيع التكراري، يمكنك الحصول على فكرة عن مقدار البيانات المتوفرة لكل متحدث.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt0.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/>
</div>

يكشف مخطط التوزيع التكراري أن حوالي ثلث المتحدثين في مجموعة البيانات لديهم أقل من 100 مثال، في حين أن حوالي عشرة متحدثين لديهم أكثر من 500 مثال. لتحسين كفاءة التدريب وتوازن مجموعة البيانات، يمكننا تحديد
البيانات إلى متحدثين يتراوح عددهم بين 100 و400 مثال.

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

دعنا نتحقق من عدد المتحدثين المتبقين:

```py
>>> len(set(dataset["speaker_id"]))
42
```

دعنا نرى عدد الأمثلة المتبقية:

```py
>>> len(dataset)
9973
```

أنت الآن لديك أقل بقليل من 10000 مثال من حوالي 40 متحدثًا فريدًا، وهو ما يجب أن يكون كافيًا.

لاحظ أن بعض المتحدثين ذوي الأمثلة القليلة قد يكون لديهم في الواقع المزيد من الصوت المتاح إذا كانت الأمثلة طويلة. ومع ذلك،
تحديد إجمالي مقدار الصوت لكل متحدث يتطلب فحص مجموعة البيانات بأكملها، وهي عملية تستغرق وقتًا طويلاً تنطوي على تحميل وفك تشفير كل ملف صوتي. لذلك، لقد اخترنا تخطي هذه الخطوة هنا.

### تضمين المتحدث

لتمكين نموذج TTS من التمييز بين متحدثين متعددين، ستحتاج إلى إنشاء تضمين متحدث لكل مثال.
تضمين المتحدث هو إدخال إضافي في النموذج الذي يلتقط خصائص صوت المتحدث.
لإنشاء هذه التضمينات المتحدث، استخدم النموذج المُدرب مسبقًا [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
من SpeechBrain.

قم بإنشاء وظيفة `create_speaker_embedding()` التي تأخذ موجة صوت إدخال وتخرج متجه 512 عنصرًا
يحتوي على تضمين المتحدث المقابل.

```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

من المهم ملاحظة أن النموذج `speechbrain/spkrec-xvect-voxceleb` تم تدريبه على الكلام باللغة الإنجليزية من مجموعة بيانات VoxCeleb
في حين أن أمثلة التدريب في هذا الدليل باللغة الهولندية. في حين أننا نعتقد أن هذا النموذج سيظل يولد
تضمينات المتحدثين المعقولين لمجموعة البيانات الهولندية الخاصة بنا، فقد لا يكون هذا الافتراض صحيحًا في جميع الحالات.

للحصول على نتائج مثالية، نوصي بتدريب نموذج X-vector على الكلام المستهدف أولاً. سيكفل ذلك قدرة النموذج على
الالتقاط بشكل أفضل لخصائص الصوت الفريدة الموجودة في اللغة الهولندية.

### معالجة مجموعة البيانات

أخيرًا، دعنا نقوم بمعالجة البيانات إلى التنسيق الذي يتوقعه النموذج. قم بإنشاء وظيفة `prepare_dataset` التي تأخذ مثالًا واحدًا وتستخدم كائن `SpeechT5Processor` لتوكينيز إدخال النص وتحميل الصوت المستهدف في مخطط Mel-spectrogram.
يجب أن تضيف أيضًا تضمينات المتحدثين كإدخال إضافي.

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # strip off the batch dimension
...     example["labels"] = example["labels"][0]

...     # use SpeechBrain to obtain x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

تحقق من صحة المعالجة عن طريق النظر في مثال واحد:

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

يجب أن تكون تضمينات المتحدثين عبارة عن متجه مكون من 512 عنصرًا:

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

يجب أن تكون التسم
لم يتم ترجمة الأجزاء المحددة من النص الأصلي بناءً على طلبك.

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # قم بتغييره إلى اسم مستودع من اختيارك
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to=["tensorboard"],
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

قم بتهيئة كائن 'Trainer' ومرر إليه النموذج ومجموعة البيانات ووسيلة تجميع البيانات.

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     tokenizer=processor,
... )
```

والآن، أنت مستعد لبدء التدريب! سيستغرق التدريب عدة ساعات. واعتمادًا على وحدة معالجة الرسوميات (GPU) لديك، فقد تواجه خطأ "نفاد الذاكرة" في CUDA عند بدء التدريب. في هذه الحالة، يمكنك تقليل `per_device_train_batch_size` بشكل تدريجي بمضاعفات 2 وزيادة `gradient_accumulation_steps` بمقدار 2x للتعويض.

```py
>>> trainer.train()
```

لاستخدام نقطة التفتيش الخاصة بك مع خط أنابيب، تأكد من حفظ المعالج مع نقطة التفتيش:

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

أرسل النموذج النهائي إلى 🤗 Hub:

```py
>>> trainer.push_to_hub()
```

## الاستنتاج

### الاستنتاج باستخدام خط أنابيب

الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستنتاج!
أولاً، دعنا نرى كيف يمكنك استخدامه مع خط أنابيب مطابق. دعنا ننشئ خط أنابيب `"text-to-speech"` مع نقطة التفتيش الخاصة بك:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

اختر قطعة من النص باللغة الهولندية التي تريد سردها، على سبيل المثال:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

لاستخدام SpeechT5 مع خط الأنابيب، ستحتاج إلى embedding للمتكلم. دعنا نحصل عليه من مثال في مجموعة الاختبار:

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

الآن يمكنك تمرير النص و embeddings للمتكلم إلى خط الأنابيب، وسيتولى بقية المهمة:

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

يمكنك بعد ذلك الاستماع إلى النتيجة:

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### تشغيل الاستدلال يدويًا

يمكنك تحقيق نفس نتائج الاستدلال دون استخدام خط الأنابيب، ولكن ستكون هناك حاجة إلى مزيد من الخطوات.

قم بتحميل النموذج من 🤗 Hub:

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

اختر مثالاً من مجموعة الاختبار للحصول على embedding للمتكلم.

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

قم بتعريف نص الإدخال وقم برمته:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

قم بإنشاء مخطط طيفي مع نموذجك:

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

قم بتصور المخطط الطيفي، إذا أردت:

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="مخطط طيفي لوغاريتمي مولد"/>
</div>

أخيرًا، استخدم المحول التناظري الرقمي لتحويل المخطط الطيفي إلى صوت.

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

من تجربتنا، قد يكون من الصعب الحصول على نتائج مرضية من هذا النموذج. تبدو جودة embeddings للمتكلم عاملاً مهمًا. نظرًا لأن SpeechT5 تم تدريبه مسبقًا باستخدام x-vectors باللغة الإنجليزية، فإنه يعمل بشكل أفضل عند استخدام embeddings للمتكلم باللغة الإنجليزية. إذا كان الكلام المُخلَّق يبدو رديئًا، فجرّب استخدام embedding مختلف للمتكلم.

ومن المرجح أيضًا أن يؤدي زيادة مدة التدريب إلى تحسين جودة النتائج. ومع ذلك، من الواضح أن الكلام باللغة الهولندية بدلاً من اللغة الإنجليزية، ويتم بالفعل التقاط خصائص صوت المتحدث (مقارنة بالصوت الأصلي في المثال).

شيء آخر يمكن تجربته هو تكوين النموذج. على سبيل المثال، جرب استخدام `config.reduction_factor = 1` لمعرفة ما إذا كان هذا يحسن النتائج.

أخيرًا، من المهم مراعاة الاعتبارات الأخلاقية. على الرغم من أن تقنية TTS لديها العديد من التطبيقات المفيدة، إلا أنها قد تُستخدم أيضًا لأغراض خبيثة، مثل انتحال صوت شخص ما دون معرفته أو موافقته. يرجى استخدام TTS بحكمة وبمسؤولية.