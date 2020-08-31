import React, { useEffect, useState } from "react";
import kuromoji from "kuromoji";
import * as tf from "@tensorflow/tfjs";
import TSNE from "tsne-js";
import skmeans from "skmeans";
import * as d3 from "d3";

const modelPromise = tf.loadLayersModel("word2vec/model.json");
const vocabularyPromise = fetch("vocabulary.json").then((response) =>
  response.json(),
);
const tokenizerPromise = new Promise((resolve) => {
  kuromoji.builder({ dicPath: "dict" }).build((_, tokenizer) => {
    resolve(tokenizer);
  });
});

const inference = async (words) => {
  const model = await modelPromise;
  const vocabulary = await vocabularyPromise;
  const weight = await model.layers[3].getWeights()[0].transpose().array();
  return words.map(({ word }) => weight[vocabulary[word]]);
};

const tokenize = async (text) => {
  const tokenizer = await tokenizerPromise;
  const vocabulary = await vocabularyPromise;
  return tokenizer
    .tokenize(text)
    .map(({ basic_form }) => basic_form)
    .filter((word) => word in vocabulary);
};

const countWords = (words) => {
  const count = {};
  for (const word of words) {
    if (!(word in count)) {
      count[word] = 0;
    }
    count[word] += 1;
  }
  return Object.entries(count).map(([word, count]) => ({ word, count }));
};

const dimensionalityReduction = (inputData) => {
  const model = new TSNE({
    dim: 2,
    perplexity: 30.0,
    earlyExaggeration: 4.0,
    learningRate: 100.0,
    nIter: 200,
    metric: "euclidean",
  });
  model.init({
    data: inputData,
    type: "dense",
  });
  model.run();
  return model.getOutputScaled();
};

const clustering = (inputData, numClusters) => {
  return skmeans(inputData, numClusters).idxs;
};

const improveLayout = (words) => {
  const simulation = d3
    .forceSimulation(words)
    .force("charge", d3.forceManyBody().strength(10))
    .force(
      "collide",
      d3
        .forceCollide()
        .radius(({ r }) => r + 1)
        .iterations(30),
    )
    .force("x", d3.forceX(0))
    .force("y", d3.forceY(0));
  simulation.tick(100).stop();
};

const visualize = async (text, { numClusters, minR, maxR }) => {
  console.time("preprocess");
  const words = countWords(await tokenize(text));
  console.timeEnd("preprocess");
  console.time("inference");
  const wordVector = await inference(words);
  console.timeEnd("inference");
  console.time("dimensionalityReduction");
  const xy = dimensionalityReduction(wordVector);
  console.timeEnd("dimensionalityReduction");
  console.time("clustering");
  const groups = clustering(xy, numClusters);
  console.timeEnd("clustering");

  const rScale = d3
    .scaleSqrt()
    .domain(d3.extent(words, ({ count }) => count))
    .range([minR, maxR]);
  words.forEach((word, i) => {
    word.x = word.x0 = xy[i][0] * 600;
    word.y = word.y0 = xy[i][1] * 600;
    word.group = groups[i];
    word.r = rScale(word.count);
  });
  console.time("improveLayout");
  improveLayout(words);
  console.timeEnd("improveLayout");
  return words;
};

const optimalFontSize = (word, r, fontFamily, fontWeight) => {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.textContent = word;
  text.setAttributeNS(null, "font-family", fontFamily);
  text.setAttributeNS(null, "font-weight", fontWeight);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.appendChild(text);
  document.body.appendChild(svg);
  let ok = 0;
  let ng = 100;
  for (let iter = 0; iter < 10; ++iter) {
    let m = (ok + ng) / 2;
    text.setAttributeNS(null, "font-size", m);
    const { width, height } = text.getBBox();
    const d = Math.sqrt(width ** 2 + height ** 2) / 2;
    if (d <= r) {
      ok = m;
    } else {
      ng = m;
    }
  }
  document.body.removeChild(svg);
  return ok;
};

const Chart = ({ words }) => {
  const contentSize = 600;
  const margin = {
    top: 10,
    right: 10,
    bottom: 10,
    left: 10,
  };
  const width = contentSize + margin.left + margin.right;
  const height = contentSize + margin.top + margin.bottom;
  const fontFamily = `'Sawarabi Gothic', sans-serif`;
  const fontWeight = "normal";

  const scale =
    contentSize /
    2 /
    Math.max(
      ...[
        Math.abs(Math.min(...words.map(({ x, r }) => x - r))),
        Math.abs(Math.max(...words.map(({ x, r }) => x + r))),
        Math.abs(Math.min(...words.map(({ y, r }) => y - r))),
        Math.abs(Math.max(...words.map(({ y, r }) => y + r))),
      ],
    );
  const color = d3.scaleOrdinal(d3.schemeCategory10);

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        <g transform={`translate(${contentSize / 2},${contentSize / 2})`}>
          {words.map((word) => {
            return (
              <g
                key={word.word}
                transform={`translate(${word.x * scale},${
                  word.y * scale
                })scale(${scale})`}
              >
                <circle r={word.r} fill={color(word.group)} />
                <text
                  fill="ghostwhite"
                  fontSize={optimalFontSize(word.word, word.r)}
                  fontFamily={fontFamily}
                  fontWeight={fontWeight}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {word.word}
                </text>
              </g>
            );
          })}
        </g>
      </g>
    </svg>
  );
};

export const App = () => {
  const [words, setWords] = useState([]);
  const defaultText = `日本国民は、正当に選挙された国会における代表者を通じて行動し、われらとわれらの子孫のために、諸国民との協和による成果と、わが国全土にわたつて自由のもたらす恵沢を確保し、政府の行為によつて再び戦争の惨禍が起ることのないやうにすることを決意し、ここに主権が国民に存することを宣言し、この憲法を確定する。そもそも国政は、国民の厳粛な信託によるものであつて、その権威は国民に由来し、その権力は国民の代表者がこれを行使し、その福利は国民がこれを享受する。これは人類普遍の原理であり、この憲法は、かかる原理に基くものである。われらは、これに反する一切の憲法、法令及び詔勅を排除する。
　日本国民は、恒久の平和を念願し、人間相互の関係を支配する崇高な理想を深く自覚するのであつて、平和を愛する諸国民の公正と信義に信頼して、われらの安全と生存を保持しようと決意した。われらは、平和を維持し、専制と隷従、圧迫と偏狭を地上から永遠に除去しようと努めてゐる国際社会において、名誉ある地位を占めたいと思ふ。われらは、全世界の国民が、ひとしく恐怖と欠乏から免かれ、平和のうちに生存する権利を有することを確認する。

われらは、いづれの国家も、自国のことのみに専念して他国を無視してはならないのであつて、政治道徳の法則は、普遍的なものであり、この法則に従ふことは、自国の主権を維持し、他国と対等関係に立たうとする各国の責務であると信ずる。

日本国民は、国家の名誉にかけ、全力をあげてこの崇高な理想と目的を達成することを誓ふ。`;
  const defaultNumClusters = 5;
  const defaultMinR = 6;
  const defaultMaxR = 30;
  useEffect(() => {
    (async () => {
      setWords(
        await visualize(defaultText, {
          minR: defaultMinR,
          maxR: defaultMaxR,
          numClusters: defaultNumClusters,
        }),
      );
    })();
  }, [defaultText, defaultNumClusters]);

  return (
    <div>
      <section className="hero is-dark">
        <div className="hero-body">
          <div className="container">
            <h1 className="title">Semantic Preserving Word Bubbles</h1>
            <h2 className="subtitle">
              As an Alternative to WordClouds for Text Visualization
            </h2>
          </div>
        </div>
      </section>
      <section className="section">
        <div className="container">
          <form
            onSubmit={async (event) => {
              event.preventDefault();
              const text = event.target.elements.text.value;
              const numClusters = +event.target.elements.numClusters.value;
              const minR = +event.target.elements.minR.value;
              const maxR = +event.target.elements.maxR.value;
              setWords(await visualize(text, { numClusters, minR, maxR }));
            }}
          >
            <div className="field">
              <label className="label">Input Text</label>
              <div className="control">
                <textarea
                  name="text"
                  className="textarea"
                  defaultValue={defaultText}
                />
              </div>
            </div>
            <div className="field">
              <label className="label">Number of Clusters</label>
              <div className="control">
                <input
                  name="numClusters"
                  type="number"
                  min="0"
                  max="10"
                  defaultValue={defaultNumClusters}
                  className="input"
                />
              </div>
            </div>
            <div className="field">
              <label className="label">Minimum Radius</label>
              <div className="control">
                <input
                  name="minR"
                  type="number"
                  min="0"
                  defaultValue={defaultMinR}
                  className="input"
                />
              </div>
            </div>
            <div className="field">
              <label className="label">Maximum Radius</label>
              <div className="control">
                <input
                  name="maxR"
                  type="number"
                  min="0"
                  defaultValue={defaultMaxR}
                  className="input"
                />
              </div>
            </div>
            <div className="field">
              <div className="control">
                <button className="button is-dark" type="submit">
                  Visualize
                </button>
              </div>
            </div>
          </form>
        </div>
      </section>
      <section className="section">
        <div className="container">
          <Chart words={words} />
        </div>
      </section>
      <footer className="footer">
        <div className="content has-text-centered">
          <p>
            <a
              href="https://github.com/likr-sandbox/wordbubble"
              target="_blank"
              rel="noopener noreferrer"
            >
              view source code
            </a>
          </p>
          <p>&copy; 2020 Yosuke Onoue</p>
        </div>
      </footer>
    </div>
  );
};
