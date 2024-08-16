# Trainer

توفر فئة [`Trainer`] واجهة برمجة تطبيقات (API) للتدريب الكامل الميزات في PyTorch، وهي تدعم التدريب الموزع على وحدات معالجة الرسوميات (GPUs) / وحدات معالجة الدقة الفائقة (TPUs) متعددة، والدقة المختلطة لوحدات معالجة الرسوميات (GPUs) من NVIDIA وAMD، و[`torch.amp`] لـ PyTorch. وتعمل فئة [`Trainer`] جنبًا إلى جنب مع فئة [`TrainingArguments`]، والتي توفر مجموعة واسعة من الخيارات لتخصيص كيفية تدريب النموذج. معًا، توفر هاتان الفئتان واجهة برمجة تطبيقات (API) تدريبًا كاملاً.

[`Seq2SeqTrainer`] و [`Seq2SeqTrainingArguments`] يرثان من فئات [`Trainer`] و [`TrainingArgument`]، وقد تم تكييفهما لتدريب النماذج الخاصة بمهام التسلسل إلى تسلسل مثل الملخص أو الترجمة.

<Tip warning={true}>
تمت تهيئة فئة [`Trainer`] لتحقيق الأداء الأمثل مع نماذج 🤗 Transformers ويمكن أن يكون لها سلوكيات مفاجئة عند استخدامها مع نماذج أخرى. عند استخدامها مع نموذجك الخاص، تأكد من:

- أن النموذج الخاص بك يعيد دائمًا الرباعيات أو الفئات الفرعية لـ [`~utils.ModelOutput`]
- يمكن للنموذج الخاص بك حساب الخسارة إذا تم توفير وسيط `labels` وأن الخسارة يتم إرجاعها كأول عنصر في الرباعية (إذا كان النموذج الخاص بك يعيد الرباعيات)
- يمكن للنموذج الخاص بك قبول وسيطات تسمية متعددة (استخدم `label_names` في [`TrainingArguments`] للإشارة إلى اسمها إلى [`Trainer`]) ولكن لا ينبغي تسمية أي منها باسم "التسمية"
</Tip>

## Trainer [[api-reference]]

[[autodoc]] Trainer

- الكل

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer

- تقييم
- التنبؤ

## TrainingArguments

[[autodoc]] TrainingArguments

- الكل

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments

- الكل