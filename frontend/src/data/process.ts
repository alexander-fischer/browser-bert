import BertModel from "../bert/model"

export function processSpamCsv(df: any, model: BertModel) {
    const data: any[] = df.data
    const inputs: string[] = data.map((row) => {
        return row["Message"]
    })

    const labels = data.map((row) => {
        const type = row["Type"]
        const label = type === "spam" ? [1] : [0]
        return label
    })

    const processedModelInputs = model.batchPreprocess(inputs, labels)
    return processedModelInputs
}
