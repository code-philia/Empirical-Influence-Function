import { useMemo, useState } from "react";
import { HighlightedTokenCodeBlock } from "./HighlightedTokenCodeBlock";

export interface SwitchTokenCodeBlockProps {
    tokens: string[],
    salienciesByToken: { [key: number]: number[] },
    answerStartIndex: number
}

export function SwitchTokenCodeBlock(props: SwitchTokenCodeBlockProps) {
    const { tokens, salienciesByToken, answerStartIndex } = props;

    const [currentIndex, setCurrentIndex] = useState(answerStartIndex);
    const convertedSaliencies = useMemo(() => convertSaliencies(salienciesByToken), [salienciesByToken]);

    return (
        <HighlightedTokenCodeBlock
            tokens={tokens}
            saliencies={convertedSaliencies[currentIndex]}
            tokenTypes={{
                [currentIndex]: 'target'   // other token types will be auto set in HighlightedTokenCodeBlock
            }}
            isTokenClickable={(i) => i >= answerStartIndex}
            onClickToken={(i) => setCurrentIndex(i)}
        />
    )
}

function convertSaliencies(salienciesByToken: { [key: number]: number[] }) {
    const converted: { [key: number]: { [key: number]: number } } = {};
    Object.entries(salienciesByToken).forEach(([idx, saliencyList]) => {
        const saliencyObject: { [key: number]: number } = {};
        saliencyList.forEach((s, i) => {
            saliencyObject[i] = s;
        })
        converted[parseInt(idx)] = saliencyObject;
    })
    return converted;
}
