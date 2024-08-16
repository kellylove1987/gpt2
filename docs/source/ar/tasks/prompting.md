# دليل إرشادي لاستخدام نماذج اللغة الضخمة

[[open-in-colab]]

نماذج اللغة الضخمة مثل Falcon و LLaMA، إلخ، هي نماذج محول مُدربة مسبقًا تم تدريبها في البداية للتنبؤ بالرمز التالي بناءً على نص الإدخال. وعادة ما تحتوي على مليارات من المعلمات وتم تدريبها على تريليونات من الرموز لفترة طويلة من الوقت. ونتيجة لذلك، تصبح هذه النماذج قوية ومتعددة الاستخدامات، ويمكنك استخدامها لحل العديد من مهام معالجة اللغات الطبيعية خارج الصندوق من خلال توجيه النماذج باستخدام موجهات اللغة الطبيعية.

غالبًا ما يُطلق على تصميم هذه الموجهات لضمان الإخراج الأمثل "هندسة الموجهات". وهندسة الموجهات هي عملية تكرارية تتطلب قدرًا كبيرًا من التجريب. اللغات الطبيعية أكثر مرونة وتعبيرية من لغات البرمجة، ومع ذلك، يمكن أن تُدخل أيضًا بعض الغموض. وفي الوقت نفسه، فإن الموجهات في اللغة الطبيعية حساسة للغاية للتغييرات. حتى التعديلات الطفيفة في الموجهات يمكن أن تؤدي إلى نتائج مختلفة تمامًا.

في حين لا توجد وصفة دقيقة لإنشاء موجهات مطابقة لجميع الحالات، فقد توصل الباحثون إلى عدد من أفضل الممارسات التي تساعد على تحقيق نتائج مثالية بشكل أكثر اتساقًا.

يغطي هذا الدليل أفضل ممارسات هندسة الموجهات لمساعدتك في صياغة موجهات أفضل لنموذج اللغة الضخمة وحل مختلف مهام معالجة اللغات الطبيعية. ستتعلم ما يلي:

- [أساسيات الإرشاد](#أساسيات-الإرشاد)
- [أفضل ممارسات إرشاد نماذج اللغة الضخمة](#أفضل-ممارسات-إرشاد-نماذج-اللغة-الضخمة)
- [تقنيات الإرشاد المتقدمة: الإرشاد القائم على القلة وسلسلة الأفكار](#تقنيات-الإرشاد-المتقدمة)
- [متى يتم الضبط الدقيق بدلاً من الإرشاد](#الإرشاد-مقابل-الضبط-الدقيق)

<Tip>

هندسة الموجهات هي مجرد جزء من عملية تحسين إخراج نموذج اللغة الضخمة. والمكون الأساسي الآخر هو اختيار إستراتيجية توليد النص المثلى. يمكنك تخصيص كيفية اختيار نموذج اللغة الضخمة الخاص بك لكل رمز من الرموز اللاحقة عند توليد النص دون تعديل أي من المعلمات القابلة للتدريب. من خلال ضبط معلمات توليد النص، يمكنك تقليل التكرار في النص المولد وجعله أكثر تماسكًا وأكثر تشابهًا مع اللغة البشرية.
تخرج إستراتيجيات ومعلمات توليد النص عن نطاق هذا الدليل، ولكن يمكنك معرفة المزيد حول هذه الموضوعات في الأدلة التالية:

* [التوليد باستخدام نماذج اللغة الضخمة](../llm_tutorial)
* [إستراتيجيات توليد النص](../generation_strategies)

</Tip>

## أساسيات الإرشاد

### أنواع النماذج

تعد معظم نماذج اللغة الضخمة الحديثة محولات فك تشفير فقط. وفيما يلي بعض الأمثلة: [LLaMA](../model_doc/llama)، [Llama2](../model_doc/llama2)، [Falcon](../model_doc/falcon)، [GPT2](../model_doc/gpt2). ومع ذلك، قد تصادف أيضًا نماذج محول الترميز والفك تشفير، على سبيل المثال، [Flan-T5](../model_doc/flan-t5) و [BART](../model_doc/bart).

تُستخدم نماذج أسلوب الترميز والفك تشفير عادةً في المهام التوليدية حيث يعتمد الإخراج **بشكل كبير** على الإدخال، على سبيل المثال، في الترجمة والتلخيص. وتُستخدم نماذج فك التشفير فقط لجميع الأنواع الأخرى من المهام التوليدية.

عند استخدام خط أنابيب لتوليد نص باستخدام نموذج لغة ضخمة، من المهم معرفة نوع نموذج اللغة الضخمة الذي تستخدمه، لأنها تستخدم خطوط أنابيب مختلفة.

قم بتشغيل الاستدلال باستخدام نماذج فك التشفير فقط باستخدام خط أنابيب `text-generation`:

```python
>>> from transformers import pipeline
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT

>>> generator = pipeline('text-generation', model = 'openai-community/gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)
[{'generated_text': "Hello, I'm a language model programmer so you can use some of my stuff. But you also need some sort of a C program to run."}]
```

لتشغيل الاستدلال باستخدام الترميز والفك تشفير، استخدم خط أنابيب `text2text-generation`:

```python
>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)
[{'generated_text': 'Je suis très heureuse de vous rencontrer.'}]
```

### النماذج الأساسية مقابل نماذج التعليمات/الدردشة

تأتي معظم نقاط التحقق الحديثة لنموذج اللغة الضخمة المتوفرة على 🤗 Hub في إصدارين: الأساسي والتعليمات (أو الدردشة). على سبيل المثال،
[`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b) و [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct).

النماذج الأساسية ممتازة في استكمال النص عند إعطائها موجهًا أوليًا، ومع ذلك، فهي ليست مثالية لمهام معالجة اللغات الطبيعية حيث يتعين عليها اتباع التعليمات، أو للاستخدام المحادثي. وهنا يأتي دور إصدارات التعليمات (الدردشة). تمثل هذه نقاط التحقق نتيجة الضبط الدقيق الإضافي للإصدارات الأساسية المُدربة مسبقًا على مجموعات بيانات التعليمات وبيانات المحادثة. يجعل هذا الضبط الدقيق الإضافي منها خيارًا أفضل للعديد من مهام معالجة اللغات الطبيعية.

دعونا نوضح بعض الموجهات البسيطة التي يمكنك استخدامها مع [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)
لحل بعض مهام معالجة اللغات الطبيعية الشائعة.

### مهام معالجة اللغات الطبيعية

أولاً، دعونا نقوم بإعداد البيئة:

```bash
pip install -q transformers accelerate
```

بعد ذلك، قم بتحميل النموذج باستخدام خط الأنابيب المناسب (`"text-generation"`):

```python
>>> from transformers import pipeline, AutoTokenizer
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipe = pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
```

<Tip>

لاحظ أن نماذج Falcon تم تدريبها باستخدام نوع بيانات `bfloat16`، لذلك نوصي باستخدام نفس النوع. يتطلب هذا إصدارًا حديثًا من CUDA ويعمل بشكل أفضل على البطاقات الحديثة.

</Tip>

الآن بعد أن قمنا بتحميل النموذج عبر خط الأنابيب، دعونا نستكشف كيف يمكنك استخدام الموجهات لحل مهام معالجة اللغات الطبيعية.

#### تصنيف النص

يعد تحليل المشاعر أحد أكثر أشكال تصنيف النص شيوعًا، والذي يقوم بتعيين علامة مثل "إيجابي" أو "سلبي" أو "محايد" إلى تسلسل نصي. دعونا نكتب موجهًا يوجه النموذج لتصنيف نص معين (مراجعة فيلم). سنبدأ بإعطاء التعليمات، ثم تحديد النص لتصنيفه. لاحظ أنه بدلاً من الاكتفاء بذلك، نقوم أيضًا بإضافة بداية الاستجابة - `"Sentiment: "`:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Classify the text into neutral, negative or positive. 
... Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
... Sentiment:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

وكنتيجة لذلك، يحتوي الإخراج على تصنيف العلامات من القائمة التي قدمناها في التعليمات، وهو تصنيف صحيح!

<Tip>

قد تلاحظ أنه بالإضافة إلى الموجه، نقوم بتمرير معلمة `max_new_tokens`. يتحكم هذا في عدد الرموز التي يجب على النموذج توليدها، وهو أحد العديد من معلمات توليد النص التي يمكنك معرفتها
في دليل [إستراتيجيات توليد النص](../generation_strategies).

</Tip>

#### التعرف على الكيانات المسماة

التعرف على الكيانات المسماة (NER) هي مهمة العثور على كيانات مسماة في قطعة من النص، مثل شخص أو موقع أو منظمة.
دعونا نقوم بتعديل التعليمات في الموجه لجعل نموذج اللغة الضخمة يؤدي هذه المهمة. هنا، دعونا نقوم أيضًا بتعيين `return_full_text = False`
بحيث لا يحتوي الإخراج على الموجه:

```python
>>> torch.manual_seed(1) # doctest: +IGNORE_RESULT
>>> prompt = """Return a list of named entities in the text.
... Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
... Named entities:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=15,
...     return_full_text = False,    
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
- Golden State Warriors
- San Francisco
```

كما ترى، قام النموذج بتحديد كيانين مسميين بشكل صحيح من النص المعطى.

#### الترجمة

يمكن أن تؤدي نماذج اللغة الضخمة مهمة الترجمة أيضًا. يمكنك اختيار استخدام نماذج الترميز والفك تشفير لهذه المهمة، ومع ذلك، هنا،
من أجل بساطة الأمثلة، سنواصل استخدام Falcon-7b-instruct، الذي يؤدي مهمة جيدة. مرة أخرى، إليك كيفية
يمكنك كتابة موجه أساسي لتوجيه النموذج لترجمة قطعة من النص من الإنجليزية إلى الإيطالية:

```python
>>> torch.manual_seed(2) # doctest: +IGNORE_RESULT
>>> prompt = """Translate the English text to Italian.
... Text: Sometimes, I've believed as many as six impossible things before breakfast.
... Translation:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=20,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
A volte, ho creduto a sei impossibili cose prima di colazione.
```

هنا أضفنا `do_sample=True` و `top_k=10` للسماح للنموذج بالمزيد من المرونة عند توليد الإخراج.

#### تلخيص النص

على غرار الترجمة، يعد تلخيص النص مهمة توليدية أخرى يعتمد فيها الإخراج **بشكل كبير** على الإدخال،
وقد تكون نماذج الترميز والفك تشفير خيارًا أفضل. ومع ذلك، يمكن استخدام نماذج أسلوب فك التشفير لهذه المهمة أيضًا.
في السابق، وضعنا التعليمات في بداية الموجه. ومع ذلك، قد يكون نهاية الموجه
مكانًا مناسبًا أيضًا للتعليمات. عادة، من الأفضل وضع التعليمات في أحد الطرفين.

```python
>>> torch.manual_seed(3) # doctest: +IGNORE_RESULT
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
... Write a summary of the above text.
... Summary:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
Permaculture is an ecological design mimicking natural ecosystems to meet basic needs and prepare for climate change. It is based on traditional knowledge and scientific understanding.
```

#### الإجابة على الأسئلة

بالنسبة لمهمة الإجابة على الأسئلة، يمكننا هيكلة الموجه إلى المكونات المنطقية التالية: التعليمات، والسياق، والسؤال،
وكلمة أو عبارة رائدة ("الإجابة:") لتوجيه النموذج لبدء توليد الإجابة:

```python
>>> torch.manual_seed(4) # doctest: +IGNORE_RESULT
>>> prompt = """Answer the question using the context below.
... Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
... Question: What modern tool is used to make gazpacho?
... Answer:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Modern tools often used to make gazpacho include
```

#### الاستدلال Reasoning

الاستدلال هي إحدى أصعب المهام بالنسبة لنماذج اللغة الضخمة، وغالبًا ما يتطلب تحقيق نتائج جيدة تطبيق تقنيات الإرشاد المتقدمة، مثل
[سلسلة الأفكار](#سلسلة-الأفكار).

دعونا نجرب ما إذا كان بإمكاننا جعل النموذج يستدل على مهمة حسابية بسيطة باستخدام موجه أساسي:

```python
>>> torch.manual_seed(5) # doctest: +IGNORE_RESULT
>>> prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: There are a total of 5 groups, so there are 5 x 4=20 students in the class.

```

صحيح! دعونا نزيد من التعقيد قليلاً ونرى ما إذا كان بإمكاننا الاستمرار في استخدام موجه أساسي:

```python
>>> torch.manual_seed(6) # doctest: +IGNORE_RESULT
>>> prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
The total number of muffins now is 21
```
هذا خطأ، الإجابة الصحيحة هي 12. في هذه الحالة، قد يكون ذلك بسبب بساطة النص التوجيهي المُعطى، أو بسبب اختيار النموذج، ففي النهاية اخترنا الإصدار الأصغر من Falcon. إن الاستدلال صعب على النماذج من جميع الأحجام، ولكن من المرجح أن تؤدي النماذج الأكبر حجمًا أداءً أفضل.

## أفضل الممارسات في توجيه LLM

في هذا القسم من الدليل، قمنا بتجميع قائمة بأفضل الممارسات التي تميل إلى تحسين نتائج النص التوجيهي:

* عند اختيار النموذج للعمل معه، من المرجح أن تؤدي أحدث النماذج والأكثر قدرة أداءً أفضل.
* ابدأ بنص توجيهي بسيط وقصير، وقم بالتكرار من هناك.
* ضع التعليمات في بداية النص التوجيهي، أو في النهاية تمامًا. عند العمل مع سياق كبير، تطبق النماذج تحسينات مختلفة لمنع تعقيد الانتباه من التصاعد التربيعي. قد يجعل هذا النموذج أكثر انتباهاً لبداية أو نهاية نص توجيهي أكثر من الوسط.
* قم بوضوح بفصل التعليمات عن النص الذي تنطبق عليه - المزيد حول هذا في القسم التالي.
* كن محددًا وواضحًا بشأن المهمة والنتيجة المرجوة - تنسيقها وطولها وأسلوبها ولغتها، وما إلى ذلك.
* تجنب الأوصاف والتعليمات الغامضة.
* يفضل استخدام التعليمات التي تقول "ماذا تفعل" بدلاً من تلك التي تقول "ماذا لا تفعل".
* "قم بتوجيه" الإخراج في الاتجاه الصحيح عن طريق كتابة الكلمة الأولى (أو حتى البدء في الجملة الأولى للنموذج).
* استخدم التقنيات المتقدمة مثل [التوجيه باستخدام القليل من الأمثلة](#few-shot-prompting) و [سلسلة الأفكار](#chain-of-thought)
* اختبر نصوصك التوجيهية مع نماذج مختلفة لتقييم متانتها.
* قم بإصدار نصوصك التوجيهية وتعقب أدائها.

## تقنيات النصوص التوجيهية المتقدمة

### التوجيه باستخدام القليل من الأمثلة

إن النصوص التوجيهية الأساسية في الأقسام أعلاه هي أمثلة على النصوص التوجيهية "ذات الصفر مثال"، مما يعني أنه تم إعطاء النموذج تعليمات وسياق، ولكن بدون أمثلة مع حلول. تعمل نماذج اللغة الكبيرة عادةً بشكل جيد على مهام "ذات الصفر مثال" هذه لأنها تم ضبطها دقيقًا على مجموعات بيانات التع��يمات. ومع ذلك، فقد تجد أن مهمتك تحتوي على المزيد من التعقيدات أو الدقائق، وربما لديك بعض المتطلبات للإخراج الذي لا يلتقطه النموذج فقط من التعليمات. في هذه الحالة، يمكنك
تجربة تقنية تسمى التوجيه باستخدام القليل من الأمثلة.

في التوجيه باستخدام القليل من الأمثلة، نقدم أمثلة في النص التوجيهي لإعطاء النموذج سياقًا إضافيًا لتحسين الأداء.
توجه الأمثلة النموذج لتوليد الإخراج باتباع الأنماط في الأمثلة.

فيما يلي مثال:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
... Date: 04/12/1961
... Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
... Date:"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=8,
...     do_sample=True,
...     top_k=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

في مقتطف الكود أعلاه، استخدمنا مثالًا واحدًا لتوضيح الإخراج المرغوب إلى النموذج، لذا يمكن تسميته
"التوجيه باستخدام مثال واحد". ومع ذلك، اعتمادًا على تعقيد المهمة، قد تحتاج إلى استخدام أكثر من مثال واحد.

قيود تقنية التوجيه باستخدام القليل من الأمثلة:
- في حين أن نماذج اللغة الكبيرة يمكن أن تلتقط الأنماط في الأمثلة، إلا أن هذه التقنية لا تعمل بشكل جيد على مهام الاستدلال المعقدة
- يتطلب التوجيه باستخدام القليل من الأمثلة إنشاء نصوص توجيهية طويلة. قد تزيد النصوص التوجيهية التي تحتوي على عدد كبير من الرموز من الحساب والزمن. هناك أيضًا حد لطول النصوص التوجيهية.
- في بعض الأحيان، عندما يتم إعطاء النموذج عددًا من الأمثلة، يمكنه أن يتعلم أنماطًا لم تكن تقصد منه تعلمها، على سبيل المثال، أن مراجعة الفيلم الثالثة سلبية دائمًا.

### سلسلة الأفكار

سلسلة الأفكار (CoT) هي تقنية تدفع النموذج إلى إنتاج خطوات استدلال وسيطة وبالتالي تحسين
النتائج على مهام الاستدلال المعقدة.

هناك طريقتان لتوجيه النموذج لإنتاج خطوات الاستدلال:
- التوجيه باستخدام القليل من الأمثلة عن طريق توضيح الأمثلة مع إجابات مفصلة للأسئلة، وإظهار النموذج كيفية حل مشكلة ما.
- عن طريق توجيه النموذج للاستدلال عن طريق إضافة عبارات مثل "دعنا نفكر خطوة بخطوة" أو "خذ نفسًا عميقًا وحل المشكلة خطوة بخطوة".

إذا طبقنا تقنية CoT على مثال الكعك المُخبوز من قسم [الاستدلال](#reasoning) واستخدمنا نموذجًا أكبر، مثل (`tiiuae/falcon-180B-chat`) الذي يمكنك تجربته في [HuggingChat](https://huggingface.co/chat/)،
سنحصل على تحسن كبير في نتيجة الاستدلال:

```text
Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
Therefore, you now have 12 muffins.


دعونا نمر خلال هذا خطوة بخطوة:

1. لديك 15 كعكة مُخبوزة.
2. أنت تأكل 2 كعكة، مما يترك لك 13 كعكة.
3. تعطي 5 كعكات لجارك، مما يترك لك 8 كعكات.
4. يشتري شريكك 6 كعكات أخرى، مما يجعل العدد الإجمالي للكعكات 14.
5. يأكل شريكك 2 كعكة، مما يترك لك 12 كعكة.
لذلك، لديك الآن 12 كعكة.
```

## المقارنة بين التحسين الدقيق والضبط الدقيق

يمكنك تحقيق نتائج رائعة من خلال تحسين موجهاتك، ولكن قد تتساءل عما إذا كان ضبط نموذج ما بدقة سيعطي نتائج أفضل في حالتك. فيما يلي بعض السيناريوهات التي قد يكون فيها ضبط نموذج أصغر بدقة خيارًا مفضلًا:

- مجالك مختلف تمامًا عما تم تدريب النماذج اللغوية الكبيرة عليه، ولم يعطِ التحسين المكثف للموجهات نتائج كافية.
- تحتاج إلى أن يعمل نموذجك بشكل جيد في لغة منخفضة الموارد.
- تحتاج إلى تدريب النموذج على بيانات حساسة تخضع لتنظيم صارم.
- يتعين عليك استخدام نموذج صغير بسبب التكلفة أو الخصوصية أو البنية التحتية أو قيود أخرى.

في جميع الأمثلة أعلاه، ستحتاج إلى التأكد من أن لديك بالفعل أو يمكنك الحصول بسهولة على مجموعة بيانات خاصة بمجال معين كبيرة بما يكفي وبتكلفة معقولة لضبط نموذج ما بدقة. ستحتاج أيضًا إلى الوقت والموارد الكافية لضبط نموذج ما بدقة.

إذا لم تكن الأمثلة أعلاه تنطبق على حالتك، فقد يثبت تحسين الموجهات أنه أكثر فائدة.