import { useEffect, useState } from "react"
import BertModel from "../src/bert/model"
import { loadCsvFile } from "../src/data/load"
import { processSpamCsv } from "../src/data/process"

export default function Home() {

    const [model, setModel] = useState<BertModel>()
    const [textToClassify, setTextToClassify] = useState("")
    const [classificationResult, setClassificationResult] = useState("")

    useEffect(() => {
        loadModel()
    }, [])

    const loadModel = async () => {
        const loadedModel = new BertModel(128)
        await loadedModel.setup()

        setModel(loadedModel)
    }

    const trainModel = async () => {
        const df = await loadCsvFile("http://localhost:3000/spam.csv")
        const trainInput = processSpamCsv(df, model)
        model.train(trainInput)
    }

    const onChangeText = (e) => {
        const value = e.target.value
        setTextToClassify(value)
    }

    const classifyText = async () => {
        if (textToClassify === undefined || textToClassify === "") {
            return
        }

        const results = await model.predict(textToClassify)
        const resultText = results[0] >= 0.5 ? "Spam" : "Ham"
        console.log(results[0])
        setClassificationResult(resultText)
    }

    return (
        <div className="flex flex-col items-center justify-center min-h-screen py-2">
            <button className="bg-red-500 text-white p-2 rounded" onClick={trainModel}>Train!</button>

            <textarea className="m-4 border-black border rounded" onChange={onChangeText} value={textToClassify} />
            <button className="m-4 bg-red-500 text-white p-2 rounded" onClick={classifyText}>Predict!</button>
            <p>Classification result: {classificationResult}</p>
        </div>
    )
}
