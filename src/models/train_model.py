from simplet5 import SimpleT5

model = SimpleT5()

model.from_pretrained(model_type='t5', model_name='t5-base')

model.train(train_df=train_df,  # pandas dataframe with 2 columns: source_text & target_text
            eval_df=eval_df,  # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len=512,
            target_max_token_len=128,
            batch_size=8,
            max_epochs=5,
            use_gpu=True,
            outputdir="outputs",
            early_stopping_patience_epochs=0,
            precision=32
            )
