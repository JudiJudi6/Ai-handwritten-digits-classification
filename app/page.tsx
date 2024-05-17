"use client";

import { useEffect, useRef, useState } from "react";
import { ReactSketchCanvas, ReactSketchCanvasRef } from "react-sketch-canvas";
import * as tf from "@tensorflow/tfjs";
import NextImage from "next/image";
import Link from "next/link";

export default function Home() {
  const canvasRef = useRef<ReactSketchCanvasRef>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [result, setResult] = useState<number | null>();

  useEffect(function () {
    loadModel();

    async function loadModel() {
      const model = await tf.loadLayersModel("/model/model.json");
      setModel(model);
    }
  }, []);

  async function handlePredict() {
    const canvas = canvasRef.current;
    if (canvas && model) {
      const dataUrl = await canvas.exportImage("jpeg");
      const image = new Image();
      image.src = dataUrl;
      await image.decode();

      let tensor = tf.browser
        .fromPixels(image)
        .mean(2)
        .expandDims(0)
        .expandDims(-1)
        .resizeNearestNeighbor([28, 28])
        .mul(tf.scalar(-1))
        .add(tf.scalar(255));

      const prediction: any = model.predict(tensor);
      const predictionArray = await prediction.array();

      const predictedDigitIndex = predictionArray[0].indexOf(
        Math.max(...predictionArray[0])
      );

      setResult(predictedDigitIndex);
    }
  }

  function handleClear() {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
      setResult(null);
    }
  }

  return (
    <div className="h-full w-full bg-white p-3 py-20 flex justify-center items-center flex-col">
      <h1 className="text-3xl text-center sm:text-4xl font-medium text-orange-500 tracking-wide">
        Convolutional Neural Network Model
      </h1>
      <p className="py-4 text-center">
        By{" "}
        <Link
          href="https://github.com/JudiJudi6"
          className="hover:text-orange-500 transition-colors duration-300 font-medium tracking-wide "
        >
          Łukasz Michnik
        </Link>
      </p>
      <p className="text-center sm:w-2/3">
        Model is learned to recognize the digits 0 - 9 on a MNIST dataset. In
        order for the model to work correctly you need to write digits similar
        to those from the dataset, sample digits are on the image below.
      </p>

      <p className="py-4">
        Test it by <span className="text-orange-500">yourself</span>
      </p>
      <p className="py-4">The model is sometimes right ;)</p>
      <div className="flex flex-col gap-5 sm:flex-row justify-center sm:items-start">
        <div className="flex flex-col justify-center items-center gap-3">
          <p>
            Draw <span className="text-orange-500">here!</span>
          </p>
          <ReactSketchCanvas
            className="border-orange-500"
            ref={canvasRef}
            strokeWidth={20}
            strokeColor="black"
            canvasColor="white"
            style={{
              border: "1px solid rgb(249 115 22)",
              boxShadow: "1px 1px 3px 0px rgba(66, 68, 90, 1)",
            }}
            width="280px"
            height="280px"
          />
        </div>
        <div className="flex flex-col justify-center items-center gap-3">
          <p>
            Imitate these <span className="text-orange-500">numbers!</span>
          </p>
          <NextImage src="/mnist.jpg" alt="" width={400} height={200} />
        </div>
      </div>
      <div className="flex justify-center items-center gap-5 py-8">
        <button
          onClick={handlePredict}
          className="text-lg px-10 tracking-wide py-2 rounded-full bg-orange-500 text-white hover:bg-orange-600 transition-colors duration-300"
        >
          Predict
        </button>
        <button
          onClick={handleClear}
          className="text-lg px-10 tracking-wide py-2 rounded-full hover:bg-red-500 hover:text-white transition-colors duration-300"
        >
          Clear
        </button>
      </div>
      <div className="pb-4">
        <p className="text-3xl">Result: {result && result}</p>
      </div>
    </div>
  );
}
