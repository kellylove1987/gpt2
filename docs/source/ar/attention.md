# آليات الانتباه 

تستخدم معظم نماذج المحول (Transformer) الانتباه الكامل بحيث تكون مصفوفة الانتباه مربعة. ويمكن أن يمثل ذلك عنق زجاجة حسابيًا كبيرًا عندما تكون لديك نصوص طويلة. ويعد Longformer وReformer من النماذج التي تحاول أن تكون أكثر كفاءة وتستخدم نسخة مبعثرة من مصفوفة الانتباه لتسريع التدريب.

## انتباه LSH

يستخدم [Reformer](model_doc/reformer) انتباه LSH. في الدالة softmax(QK^t)، فإن أكبر العناصر فقط (في بعد softmax) من المصفوفة QK^t هي التي ستعطي مساهمات مفيدة. لذلك، بالنسبة لكل استعلام q في Q، يمكننا أن نأخذ في الاعتبار فقط المفاتيح k في K القريبة من q. وتُستخدم دالة هاش لتحديد ما إذا كان q وk قريبين. ويتم تعديل قناع الانتباه لحجب الرمز الحالي (باستثناء الموضع الأول)، لأنه سيعطي استعلامًا ومفتاحًا متساويين (لذلك متشابهين للغاية). نظرًا لأن الهاش يمكن أن يكون عشوائيًا بعض الشيء، يتم في الممارسة العملية استخدام عدة دوال هاش (يحددها معامل n_rounds) ثم يتم حساب المتوسط معًا.

## الانتباه المحلي

يستخدم [Longformer](model_doc/longformer) الانتباه المحلي: غالبًا ما يكون السياق المحلي (على سبيل المثال، ما هما الرمزان إلى اليسار واليمين؟) كافيًا لاتخاذ إجراء بالنسبة للرمز المعطى. أيضًا، عن طريق تكديس طبقات الانتباه التي لها نافذة صغيرة، سيكون للطبقة الأخيرة مجال استقبال أكبر من مجرد الرموز في النافذة، مما يسمح لها ببناء تمثيل للجملة بأكملها.

كما يتم منح بعض رموز الإدخال المختارة مسبقًا انتباهًا عالميًا: بالنسبة لهذه الرموز القليلة، يمكن لمصفوفة الانتباه الوصول إلى جميع الرموز وتكون هذه العملية متماثلة: فلجميع الرموز الأخرى إمكانية الوصول إلى تلك الرموز المحددة (بالإضافة إلى تلك الموجودة في نافذتهم المحلية). وهذا موضح في الشكل 2d من الورقة، انظر أدناه لمثال على قناع الانتباه:

<div class="flex justify-center">
    <img scale="50 %" align="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>

وباستخدام مصفوفات الانتباه هذه التي تحتوي على معلمات أقل، يسمح النموذج بإدخالات ذات طول تسلسل أكبر.

## حيل أخرى

### الترميزات الموضعية المحورية

يستخدم [Reformer](model_doc/reformer) ترميزات موضعية محورية: في نماذج المحول التقليدية، يكون الترميز الموضعي E مصفوفة بحجم \\(l\\) في \\(d\\)، حيث \\(l\\) هو طول التسلسل و\\(d\\) هو بعد الحالة المخفية. إذا كان لديك نصوص طويلة جدًا، فقد تكون هذه المصفوفة ضخمة وتستهلك مساحة كبيرة جدًا على وحدة معالجة الرسوميات (GPU). وللتخفيف من ذلك، تتكون الترميزات الموضعية المحورية من تحليل تلك المصفوفة الكبيرة E إلى مصفوفتين أصغر E1 وE2، بأبعاد \\(l_{1} \times d_{1}\\) و \\(l_{2} \times d_{2}\\)، بحيث \\(l_{1} \times l_{2} = l\\) و\\(d_{1} + d_{2} = d\\) (مع حاصل ضرب الأطوال، ينتهي الأمر بكونه أصغر بكثير). ويتم الحصول على الترميز للخطوة الزمنية \\(j\\) في E عن طريق ربط الترميزات للخطوة الزمنية \\(j \% l1\\) في E1 و \\(j // l1\\) في E2.