import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");

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
const classificationResultDiv = document.getElementById("classificationResult");  // Added this to display the result

const jsonButton = document.getElementById("jsonButton");
jsonButton.addEventListener("click", downloadDataAsJSON);

let yArray = [];
let mArray = [];
let cArray = [];
let aArray = [];


//json data arrays for letters
let dataY = [];
let dataM = [];
let dataC = [];
let dataA = [];






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

    enableWebcamButton.addEventListener("click", (e) => enableCam(e));
    logButton.addEventListener("click", (e) => logAllPoses(e));
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
 // LOG POSE COORDINATES
 ********************************************************************/
function logAllPoses() {
    const label = document.getElementById('rps').value;

    if (results.landmarks && results.landmarks.length > 0) {
        for (let pose of results.landmarks) {
            const flatArray = pose.flatMap(point => [point.x, point.y, point.z]);

            const poseData = {
                points: flatArray,
                label: label
            };

            if (label === "Y") {
                yArray.push(poseData);
                dataY.push(poseData);
            } else if (label === "M") {
                mArray.push(poseData);
                dataM.push(poseData);
            } else if (label === "C") {
                cArray.push(poseData);
                dataC.push(poseData);
            } else if (label === "A") {
                aArray.push(poseData);
                dataA.push(poseData);
            }

            machine.learn(flatArray, label);

            console.log(`Pose data logged for label: ${label}`);
            console.log(flatArray, poseData);
        }
    }
}

/********************************************************************
 // CLASSIFY POSE
 ********************************************************************/
function classifyHandPose(results) {
    if (results.landmarks && results.landmarks.length > 0) {
        for (let pose of results.landmarks) {
            const flatArray = pose.flatMap(point => [point.x, point.y, point.z]);
            console.log(flatArray);
            console.log(machine);
            const classification = machine.classify(flatArray);
            console.log(`The pose is classified as: ${classification}`);

            classificationResultDiv.textContent = `Predicted Pose: ${classification}`;
        }
    }
}

/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createPoseLandmarker();
}


/*****************************************************************
 // training logger zodat ik accurcy kan bereknen
     ***********************************************************/
let testSet = [];

document.getElementById("logTestData").addEventListener("click", () => {
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
});



document.getElementById('classifyButton').addEventListener('click', () => {
    classifyHandPose(results);
});




/**********
 Run die test neef
 *********/

document.getElementById("runTest").addEventListener("click", () => {
    let correct = 0;

    // bijhouden in console hoe veel en wat fout is gegaan
    const confusionLog = [];

    testSet.forEach(test => {
        const prediction = machine.classify(test.points);
        // console.log(test.points);
        // console.log(prediction);
        if (prediction === test.label) {
            correct++;
        } else {

            // als het niet correct is zet in arrya
            confusionLog.push({
                actual: test.label,
                predicted: prediction
            });
        }
    });

    const accuracy = (correct / testSet.length) * 100;
    console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

    alert(`Model accuracy: ${accuracy.toFixed(2)}% (${correct}/${testSet.length} correct)`);

    if (confusionLog.length > 0) {
        console.log("❌ Misclassifications:");
        confusionLog.forEach(entry => {
            console.log(`Actual: ${entry.actual} → Predicted: ${entry.predicted}`);
        });
    } else {
        console.log("🎉 No misclassifications! Good job!");
    }
});



/***************************************************
 // JSON DOWNLOAD FUNCTION
 **********************************************/

function downloadDataAsJSON() {

    console.log('json function');
    const data = {
        Y: dataY,
        M: dataM,
        C: dataC,
        A: dataA
    };

    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "handpose-data.json";
    a.click();

    URL.revokeObjectURL(url);
}

