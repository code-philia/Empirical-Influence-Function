import { useMemo, useState } from 'react';
import latestSaliencyJSONText from '../../../latest_saliency.json?raw';
import './App.css';
import { SwitchTokenCodeBlock } from './components/SwitchTokenCodeBlock';

const allSaliencies = JSON.parse(latestSaliencyJSONText);
// console.log(allSaliencies);

const trainSamplesSize = allSaliencies['related_train_samples'].length;

function App() {
  return (<View2 />);
}

// function View1() {
//   const [testState, setTargetTestState] = useState(0);
//   const [trainIndex, setTrainIndex] = useState(0);

//   const targetTestSample = useMemo(() => allSaliencies['target_test_sample'], []);
//   const convertedTestTokensBefore = useMemo(() => convertTokens(targetTestSample['before']['full_tokens']), []);
//   const convertedTestTokensAfter = useMemo(() => convertTokens(targetTestSample['after']['full_tokens']), []);
//   const convertedTestSaliencyBefore = useMemo(() => convertRawSaliencyToObject(targetTestSample['before']['saliency_list']), []);
//   const convertedTestSaliencyAfter = useMemo(() => convertRawSaliencyToObject(targetTestSample['after']['saliency_list']), []);

//   const trainSample = useMemo(() => allSaliencies['related_train_samples'][trainIndex], [trainIndex]);
//   const convertedTrainTokensBefore = useMemo(() => convertTokens(trainSample['before_original']['full_tokens']), [trainSample]);
//   const convertedTrainTokensAfter = useMemo(() => convertTokens(trainSample['after_original']['full_tokens']), [trainSample]);
//   const convertedTrainSaliencyBefore = useMemo(() => convertRawSaliencyToObject(trainSample['before_original']['saliency_list']), [trainSample]);
//   const convertedTrainSaliencyAfter = useMemo(() => convertRawSaliencyToObject(trainSample['after_original']['saliency_list']), [trainSample]);

//   return (
//     <>
//       <div style={{ display: 'flex', width: '90vw', margin: '0 2rem', gap: '2rem' }}>
//         <div className="demo-code-block" style={{ flex: '2 0 0' }}>
//           <div className="demo-code-block-title" style={{ display: 'flex', gap: '1rem', alignItems: 'end' }}>
//             <div style={{ fontWeight: 'bold' }}>Target Test Sample</div>
//             <div style={{ fontSize: 'small' }}>{testState === 0 ? 'true' : 'pred'}</div>
//             <div style={{ display: 'flex', gap: '0.5rem' }}>
//               <button onClick={() => setTargetTestState((testState - 1) % 2)}>last</button>
//               <button onClick={() => setTargetTestState((testState + 1) % 2)}>next</button>
//             </div>
//           </div>
//           <div className="demo-code-block-area" style={{ display: 'flex', gap: '1rem' }}>
//             <div>
//               <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Ground Truth</div>
//               <SwitchTokenCodeBlock
//                 key={testState}
//                 tokens={convertedTestTokensBefore}
//                 salienciesByToken={convertedTestSaliencyBefore}
//                 answerStartIndex={targetTestSample['before']['start_index']}
//               />
//             </div>
//             <div>
//               <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Prediction</div>
//               <SwitchTokenCodeBlock
//                 key={testState}
//                 tokens={convertedTestTokensAfter}
//                 salienciesByToken={convertedTestSaliencyAfter}
//                 answerStartIndex={targetTestSample['after']['start_index']}
//               />
//             </div>
//           </div>
//         </div>
//         <div className="demo-code-block" style={{ flex: '2 0 0' }}>
//           <div className="demo-code-block-title" style={{ display: 'flex', gap: '1rem', alignItems: 'end' }}>
//             <div style={{ fontWeight: 'bold' }}>Related Train Sample</div>
//             <div style={{ fontSize: 'small' }}>index: {trainIndex}</div>
//             <div style={{ display: 'flex', gap: '0.5rem' }}>
//               <button onClick={() => setTrainIndex((trainIndex - 1) % trainSamplesSize)}>last</button>
//               <button onClick={() => setTrainIndex((trainIndex + 1) % trainSamplesSize)}>next</button>
//             </div>
//           </div>
//           <div className="demo-code-block-area" style={{ display: 'flex', gap: '1rem' }}>
//             <div>
//               <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Before Overfitting</div>
//               <SwitchTokenCodeBlock
//                 key={trainIndex}
//                 tokens={convertedTrainTokensBefore}
//                 salienciesByToken={convertedTrainSaliencyBefore}
//                 answerStartIndex={trainSample['before_original']['start_index']}
//               />
//             </div>
//             <div>
//               <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>After Overfitting</div>
//               <SwitchTokenCodeBlock
//                 key={trainIndex}
//                 tokens={convertedTrainTokensAfter}
//                 salienciesByToken={convertedTrainSaliencyAfter}
//                 answerStartIndex={trainSample['after_original']['start_index']}
//               />
//             </div>
//           </div>
//         </div>
//       </div>
//     </>
//   )
// }

function View2() {
  const [testState, setTargetTestState] = useState(0);
  const [trainIndex, setTrainIndex] = useState(0);

  const targetTestSample = useMemo(() => allSaliencies['target_test_sample'], []);
  const convertedTestTokensBefore = useMemo(() => convertTokens(targetTestSample['before']['full_tokens']), []);
  const convertedTestTokensAfter = useMemo(() => convertTokens(targetTestSample['after']['full_tokens']), []);
  const convertedTestSaliencyBefore = useMemo(() => convertRawSaliencyToObject(targetTestSample['before']['saliency_list']), []);
  const convertedTestSaliencyAfter = useMemo(() => convertRawSaliencyToObject(targetTestSample['after']['saliency_list']), []);

  const trainSample = useMemo(() => allSaliencies['related_train_samples'][trainIndex], [trainIndex]);
  const convertedTrainTokensBefore = useMemo(() => convertTokens(trainSample['before_original']['full_tokens']), [trainSample]);
  const convertedTrainSaliencyBefore = useMemo(() => convertRawSaliencyToObject(trainSample['before_original']['saliency_list']), [trainSample]);

  // // Print all samples
  // console.log('```\n' + allSaliencies['related_train_samples'].map((x: any) => convertTokens(x['before_original']['full_tokens']).join('')).join('```\n\n```\n') + '```')

  return (
    <>
      <div style={{ display: 'flex', width: '90vw', margin: '0 2rem', gap: '2rem' }}>
        <div className="demo-code-block" style={{ flex: '2 0 0' }}>
          <div className="demo-code-block-title" style={{ display: 'flex', gap: '1rem', alignItems: 'end' }}>
            <div style={{ fontWeight: 'bold' }}>Target Test Sample</div>
          </div>
          <div className="demo-code-block-area" style={{ display: 'flex', gap: '1rem' }}>
            <div>
              <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Ground Truth</div>
              <SwitchTokenCodeBlock
                key={testState}
                tokens={convertedTestTokensBefore}
                salienciesByToken={convertedTestSaliencyBefore}
                answerStartIndex={targetTestSample['before']['start_index']}
              />
            </div>
                        <div>
              <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Prediction</div>
              <SwitchTokenCodeBlock
                key={testState}
                tokens={convertedTestTokensAfter}
                salienciesByToken={convertedTestSaliencyAfter}
                answerStartIndex={targetTestSample['before']['start_index']}
              />
            </div>
          </div>
        </div>
        <div className="demo-code-block" style={{ flex: '1 0 0' }}>
          <div className="demo-code-block-title" style={{ display: 'flex', gap: '1rem', alignItems: 'end', flexWrap: 'wrap' }}>
            <div style={{ fontWeight: 'bold' }}>Related Train Sample</div>
            <div style={{ fontSize: 'small' }}>index: {trainIndex}</div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button onClick={() => setTrainIndex((trainIndex - 1) % trainSamplesSize)}>last</button>
              <button onClick={() => setTrainIndex((trainIndex + 1) % trainSamplesSize)}>next</button>
            </div>
          </div>
          <div className="demo-code-block-area" style={{ display: 'flex', gap: '1rem' }}>
            <div>
              <div style={{ marginTop: '0.3rem', fontSize: 'small' }}>Ground Truth</div>
              <SwitchTokenCodeBlock
                key={trainIndex}
                tokens={convertedTrainTokensBefore}
                salienciesByToken={convertedTrainSaliencyBefore}
                answerStartIndex={trainSample['before_original']['start_index']}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

function convertRawSaliencyToObject(saliency: number[]) {
  const converted: { [key: number]: number[] } = {};
  saliency.forEach((x: any) => {
    const idx = x['index'];
    converted[idx] = x['saliency'];
  })
  return converted;
}

function convertTokens(tokens: string[]) {
  return tokens.map(t => t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  '));
}

export default App;
