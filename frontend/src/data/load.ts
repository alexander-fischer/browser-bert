import axios from "axios"
import { readString } from "react-papaparse"

export async function loadCsvFile(url: string) {
    try {
        const res = await axios.get(url)
        const parsedCsv = readString(res.data, { header: true })
        return parsedCsv
    } catch (e) {
        console.error(e)
        return
    }
}
