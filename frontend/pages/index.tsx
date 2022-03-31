import { useEffect, useState } from "react"
import BertModel from "../src/bert/model"
import { processSpamCsv } from "../src/data/process"
import { usePapaParse } from "react-papaparse"
import axios from "axios"

export default function Home() {
    const { readString } = usePapaParse()

    const [model, setModel] = useState<BertModel>()
    const [data, setData] = useState<unknown>()
    const [textToClassify, setTextToClassify] = useState("spam spam spam free free free")
    const [classificationResult, setClassificationResult] = useState("")
    const [modelTrainState, setModelTrainingState] = useState<ModelTrainingState>(ModelTrainingState.NOT_TRAINED)

    useEffect(() => {
        loadModel()
        loadData()
    }, [])

    const loadModel = async () => {
        const loadedModel = new BertModel(128)
        await loadedModel.setup()

        setModel(loadedModel)
    }

    const trainModel = async () => {
        setModelTrainingState(ModelTrainingState.TRAINING)

        const trainInput = processSpamCsv(data, model)
        await model.train(trainInput)

        setModelTrainingState(ModelTrainingState.TRAINED)
    }

    const onChangeText = (e) => {
        const value = e.target.value
        setTextToClassify(value)
    }

    const loadData = async () => {
        const res = await axios.get("/spam.csv")
        readString(res.data, {
            worker: true,
            header: true,
            complete: (results) => {
                console.log("Data loaded")
                setData(results)
            },
        })
    };


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
            {
                modelTrainState === ModelTrainingState.NOT_TRAINED && <button className="bg-red-500 text-white p-2 rounded w-20" onClick={trainModel}>Train!</button>
            }

            {
                modelTrainState === ModelTrainingState.TRAINING && <button className="bg-red-500 text-white p-2 rounded">Training runs...</button>
            }

            {
                modelTrainState === ModelTrainingState.TRAINED && <>
                    <textarea className="border-black border rounded" onChange={onChangeText} value={textToClassify} />
                    <button className="mt-2 mb-4 bg-red-500 text-white p-2 rounded" onClick={classifyText}>Predict!</button>
                    <p>Classification result: {classificationResult}</p>
                </>
            }

        </div>
    )
}

enum ModelTrainingState {
    NOT_TRAINED = "not trained",
    TRAINING = "training",
    TRAINED = "trained"
}
