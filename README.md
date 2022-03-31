# Browser BERT

Use BERT for transfer learning - but just in the browser. Find the article [here](https://alexfi.dev/blog/tensorflowjs-bert-train).

## Setup BERT

Install dependencies

`poetry install`

Setup BERT model

`poetry run setup_bert`

## Setup Frontend

Install dependencies

`cd frontend && yarn install`

Convert model

`yarn convert_model`

Run frontend

`yarn dev`

## Demo

https://browser-bert.vercel.app/

## Dataset

Can be found [here](https://github.com/bigmlcom/python/blob/master/data/spam.csv).