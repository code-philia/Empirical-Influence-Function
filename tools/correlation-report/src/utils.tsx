export const scaleToUnit = (
    values: readonly number[],
    options: { gamma?: number; eps?: number } = {}
): number[] => {
    const gamma = options.gamma ?? 0.5; // < 1 amplifies differences
    const eps = options.eps ?? 1e-12;

    // 1) Min-max normalize
    let min = Infinity;
    let max = -Infinity;
    for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const range = Math.max(max - min, eps);

    // 2) Apply gamma to sharpen
    return values.map((v) => {
        const x = (v - min) / range;
        return Math.pow(x, gamma);
    });
};