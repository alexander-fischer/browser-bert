import * as tf from "@tensorflow/tfjs"
import { FRONTEND_URL } from "../constants"
import { BertTokenizer, loadTokenizer } from "./tokenizer"

export default class BertModel {
    public inputSize: number
    public url = FRONTEND_URL + "/model/model.json"

    public bertModel: tf.GraphModel
    public tokenizer: BertTokenizer
    public model: tf.Sequential

    constructor(inputSize: number) {
        this.inputSize = inputSize
    }

    public async setup() {
        const setupCalls: Promise<void>[] = []

        if (this.model === undefined) {
            setupCalls.push(this.loadBertModel())
        }

        if (this.tokenizer === undefined) {
            setupCalls.push(this.loadTokenizer())
        }

        try {
            await Promise.all(setupCalls)
            console.log(`Setup completed`)
        } catch (e) {
            console.log(`Setup error: ${e}`)
        }
    }

    public preprocess(input: string) {
        const inputs: string[] = [input]
        const processedInputs = this.batchPreprocess(inputs)
        return processedInputs[0]
    }

    // Preprocess dataset
    public batchPreprocess(inputExamples: string[], inputLabels?: number[][]) {
        const tokenizedInputs = inputExamples.map((input) =>
            this.tokenizer.encodeText(input, this.inputSize)
        )

        const bertInputs: BertInput[] = tokenizedInputs.map(
            (tokenized, index) => {
                const bertInput: BertInput = {
                    inputIds: tokenized.inputIds,
                    inputMask: tokenized.inputMask,
                    segmentIds: tokenized.segmentIds,
                    labels: inputLabels?.[index],
                }
                return bertInput
            }
        )

        return bertInputs
    }

    public async train(inputs: BertInput[], batchSize = 32) {
        console.log("Start training...")

        const bertOutput = await this.bertLayerInference(inputs)
        const x = tf.tensor2d(
            bertOutput,
            [inputs.length, this.inputSize * this.inputSize],
            "int32"
        )

        const labels = inputs.map((input) => input.labels)
        const y = tf.tensor2d(labels, [inputs.length, 1], "int32")

        const model = this.createClassificationLayer()
        const history = await model.fit(x, y, {
            batchSize,
            epochs: 10,
            verbose: 1,
        })
        console.log(
            `Trained with accuracy of: ${
                history.history.acc[history.history.acc.length - 1]
            }`
        )

        this.model = model
    }

    public async predict(inputText: string) {
        const processedInput = this.preprocess(inputText)
        const predictions = await this.batchPredict([processedInput])
        return predictions[0]
    }

    public async batchPredict(inputs: BertInput[]) {
        const bertOutput = await this.bertLayerInference(inputs)
        const x = tf.tensor2d(
            bertOutput,
            [inputs.length, this.inputSize * this.inputSize],
            "int32"
        )

        const predTensor = this.model.predict(x) as tf.Tensor2D
        const predictions = await predTensor.array()
        return predictions
    }

    // Get raw results from bert layer
    private async bertLayerInference(inputs: BertInput[]) {
        const batchSize = inputs.length
        const inputIds = inputs.map((value) => value.inputIds)
        const segmentIds = inputs.map((value) => value.segmentIds)
        const inputMask = inputs.map((value) => value.inputMask)

        const rawResult = tf.tidy(() => {
            const tfInputIds = tf.tensor2d(
                inputIds,
                [batchSize, this.inputSize],
                "int32"
            )
            const tfSegmentIds = tf.tensor2d(
                segmentIds,
                [batchSize, this.inputSize],
                "int32"
            )
            const tfInputMask = tf.tensor2d(
                inputMask,
                [batchSize, this.inputSize],
                "int32"
            )
            return this.bertModel.execute({
                input_ids: tfInputIds,
                token_type_ids: tfSegmentIds,
                attention_mask: tfInputMask,
            })
        }) as tf.Tensor2D
        const bertOutput = await rawResult.array()
        rawResult.dispose()

        return bertOutput
    }

    // Add the classification layer
    private createClassificationLayer() {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [this.inputSize * this.inputSize],
                    units: 1,
                    activation: "sigmoid",
                }),
            ],
        })

        model.compile({
            optimizer: tf.train.adam(0.0001),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"],
        })

        return model
    }

    // Load converted bert model
    private async loadBertModel() {
        console.log("Loading model...")
        this.bertModel = await tf.loadGraphModel(this.url)
        console.log("Model loaded")
    }

    // Load tokenizer for bert input
    private async loadTokenizer() {
        console.log("Loading tokenizer...")
        this.tokenizer = await loadTokenizer()
        console.log("Tokenizer loaded")
    }
}

export interface BertInput {
    inputIds: number[]
    inputMask: number[]
    segmentIds: number[]
    labels?: number[]
}
