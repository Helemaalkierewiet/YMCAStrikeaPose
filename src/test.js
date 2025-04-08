import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logTestData");  // Renamed to logTestData for the logging button
const classifyButton = document.getElementById("runTest");  // Button for running the test

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let poseLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

const k = 3;
const machine = new kNear(k);

let image = document.querySelector("#myimage");
const classificationResultDiv = document.getElementById("classificationResult");

let testSet = [];

/********************************************************************
 // CREATE THE POSE LANDMARKER
 ********************************************************************/
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    console.log("Pose model loaded. Ready!");

    enableWebcamButton.addEventListener("click", () => enableCam());
    logButton.addEventListener("click", () => logTestData());
    classifyButton.addEventListener("click", () => runTest());
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });

    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await poseLandmarker.detectForVideo(video, performance.now());

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks && results.landmarks.length > 0) {
        for (let pose of results.landmarks) {
            drawUtils.drawLandmarks(pose, { radius: 4, color: "#FF0000", lineWidth: 2 });
            drawUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });

            const rightWrist = pose[16];
            if (rightWrist) {
                image.style.transform = `translate(${rightWrist.x * video.videoWidth}px, ${rightWrist.y * video.videoHeight}px)`;
            }
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

/********************************************************************
 // LOG TEST DATA (For testing purposes)
 ********************************************************************/
function logTestData() {
    const label = document.getElementById("testLabel").value;

    if (results.landmarks && results.landmarks.length > 0) {
        for (let pose of results.landmarks) {
            const flatArray = pose.flatMap(p => [p.x, p.y, p.z]);

            testSet.push({
                label: label,
                points: flatArray
            });

            console.log(`Test data logged for ${label}`);
        }
    }
}

/********************************************************************
 // RUN THE TEST AND CALCULATE ACCURACY
 ********************************************************************/
function runTest() {
    let correct = 0;

    testSet.forEach(test => {
        const prediction = machine.classify(test.points);
        console.log(test.points);
        console.log(prediction);
        if (prediction === test.label) {
            correct++;
        }
    });

    const accuracy = (correct / testSet.length) * 100;
    console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

    alert(`Model accuracy: ${accuracy.toFixed(2)}% (${correct}/${testSet.length} correct)`);
}

/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createPoseLandmarker();
} else {
    console.error('Webcam not supported or permissions denied.');
}
