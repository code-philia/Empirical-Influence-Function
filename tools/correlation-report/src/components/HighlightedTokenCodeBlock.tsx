import React, { useEffect, useRef } from "react";
import { scaleToUnit } from "../utils";

export interface HighlightedTokenCodeBlockProps {
    tokens: string[],
    saliencies?: { [key: number]: number },
    tokenTypes?: { [key: number]: string },
    colors?: { [key: number]: string },
    isTokenClickable?: (tokenIndex: number) => boolean,
    onClickToken?: (tokenIndex: number) => void
}

const defaultColors: { [key: string]: React.CSSProperties } = {
    none: {
        color: 'black'
    },
    target: {
        fontWeight: 'bold',
        backgroundColor: '#ff000066'
    },
    behindTarget: {
        color: '#00000022'
    }
}


// OkLab color to show saliency
type OkLch = { L: number; c: number; h: number, a: number };
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const toOkLch = ({ L, c, h, a }: OkLch) => `oklch(${L} ${c} ${h} / ${a})`;

export const saliencyNormalize = (
    values: number[]
): number[] => {
    return scaleToUnit(values, { gamma: 4.0 });
}

export const saliencyToYellowOpacity = (
    saliency: number,
    lightnessRange: { min: number; max: number } = { min: 0.20, max: 1.00 }
): string => {
    // 1) Clamp saliency and map to lightness range
    const s = clamp01(saliency);
    let a = s < 1e-3 ? 0 : lerp(lightnessRange.min, lightnessRange.max, s);

    // 2) Fixed "yellow" chroma in OKLab (tuned for readability)
    const L = 0.9496;
    const c = 0.1053;
    const h = 105;

    return toOkLch({ L, c, h, a });
};

function renderToken(token: string, key?: string, score?: number, saliency?: number, tokenType?: string, color?: string, onClickToken?: () => void) {
    const baseStyle: React.CSSProperties = {};
    const baseProps: React.HTMLAttributes<HTMLSpanElement> = {};

    const style: React.CSSProperties = { ...baseStyle };
    const props: React.HTMLAttributes<HTMLSpanElement> = { ...baseProps };

    // add style
    if (color !== undefined) {
        style.color = color;
    }
    else if (tokenType !== undefined) {
        Object.assign(style, defaultColors[tokenType]);
    }
    else if (score !== undefined) {
        style.backgroundColor = saliencyToYellowOpacity(score);
    } else {
        Object.assign(style, defaultColors['none']);
    }
    
    if (saliency !== undefined) {
        props.title = `(${key}) ${saliency}`;
    }

    // add event handler
    if (onClickToken !== undefined) {
        style.cursor = 'pointer';
        props.onClick = onClickToken;
    }

    return (<span key={key} style={style} {...props}>{token}</span>);
}

export function HighlightedTokenCodeBlock(props: HighlightedTokenCodeBlockProps) {
    const { tokenTypes, colors, isTokenClickable, onClickToken } = props;
    const preRef: React.RefObject<HTMLPreElement | null> = useRef(null);

    // normalize saliencies
    let { saliencies } = props;
    
    const scores = { ...saliencies }; // necessary, use a copy
    if (scores !== undefined) {
        const _k: number[] = [];
        let _v: number[] = [];

        Object.entries(scores).forEach(([k, v]) => {
            _k.push(parseInt(k));
            _v.push(v);
        })

        _v = saliencyNormalize(_v);

        _k.forEach((k, i) => {
            scores[k] = _v[i];
        });
    }

    // gray out tokens behind target
    let targetTokenIndex: number | undefined;
    if (tokenTypes !== undefined) {
        Object.entries(tokenTypes).forEach(([k, v]) => {
            if (v === 'target') {
                targetTokenIndex = parseInt(k);
            }
        })
    }

    useEffect(() => {
        if (preRef.current) {
            preRef.current.scrollTop = preRef.current.scrollHeight;
        }
    }, []);

    return (
        <pre className="highlighted-token-code-block" ref={preRef}>
            <code>
                {props.tokens.map((v, i) => {
                    const tokenType = targetTokenIndex !== undefined && i > targetTokenIndex ? 'behindTarget' : tokenTypes?.[i];
                    const onClickCurrentToken = isTokenClickable?.(i) ? () => onClickToken?.(i) : undefined;
                    return renderToken(v, `${i}`, scores?.[i], saliencies?.[i], tokenType, colors?.[i], onClickCurrentToken);
                })}
            </code>
        </pre>
    )
}