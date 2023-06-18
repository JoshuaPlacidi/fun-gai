import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";
import gaussian from "gaussian";

import ImageCanvas from "../components/ImageCanvas";
import XYPlot from "../components/XYPlot";
import DisplayImage from "../components/DisplayImage";
import SpeciesList from "../components/SpeciesList";
import Explanation from "../components/Explanation";

import axios from 'axios';


import { rounder } from "../utils";

import "./App.css";

import encodedData from "../encoded.json";

const MODEL_PATH = "http://127.0.0.1:8080/front-end/public/models/generatorjs/model.json";

class App extends Component {
  constructor(props) {
    super(props);
    this.getImage = this.getImage.bind(this);

    this.norm = gaussian(0, 1);

    this.state = {
      model: null,
      digitImg: tf.zeros([28, 28]),
      mu: 0,
      sigma: 0
    };
  }

  componentDidMount() {
    tf
      .loadModel(MODEL_PATH)
      .then(model => this.setState({ model }))
      .then(() => this.getImage())
      .then(digitImg => this.setState({ digitImg }));
  }

  async getImage() {
    const { model, mu, sigma } = this.state;
    const zSample = [mu, sigma];

    const response = fetch('http://localhost:9999/decode', {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        zSample: zSample
      }),
    }).then(res => res.json()).then(res => {console.log(res);
      tensor = torch.tensor(image_list)
    print(tensor.shape)
    });

    



    // return model
    //   .predict(zSample)
    //   .mul(tf.scalar(255.0))
    //   .reshape([28, 28]);
  }

  render() {
    return (
      <div className="App">
        <h1>VAE Latent Space Explorer</h1>
        <div className="DisplayImage">
          <DisplayImage />
        </div>
        <div className="SpeciesList">
          <SpeciesList />
        </div>
        <div className="ImageDisplay">
          <ImageCanvas
            width={500}
            height={500}
            imageData={this.state.digitImg}
          />
        </div>

        <div className="ChartDisplay">
          <XYPlot
            data={encodedData}
            width={500 - 10 - 10}
            height={500 - 20 - 10}
            xAccessor={d => d[0]}
            yAccessor={d => d[1]}
            colorAccessor={d => d[2]}
            margin={{ top: 20, bottom: 10, left: 10, right: 10 }}
            onHover={({ x, y }) => {
              this.setState({ sigma: y, mu: x });
              this.getImage().then(digitImg => this.setState({ digitImg }));
            }}
          />
        </div>
        <p>Mu: {rounder(this.norm.cdf(this.state.mu), 3)}</p>
        <p>Sigma: {rounder(this.norm.cdf(this.state.sigma), 3)}</p>
      </div>
    );
  }
}

export default App;
