import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

let poseLandmarker, webcamRunning = false, results = null;
const k = 3;
const machine = new kNear(k);

const image = document.querySelector("#myimage");
const posePrompt = document.getElementById("posePrompt");
const feedback = document.getElementById("feedback");


const checkmark = document.getElementById("checkmark");
const timerElement = document.getElementById("timer");
let timerCountdown = null;
// Game state
let currentPose = null;
let lastPoseChangeTime = 0;
const poseOptions = ["Y", "M", "C", "A"];
const poseChangeInterval = 5000; // Change pose every 5 seconds

document.getElementById("startButton").addEventListener("click", async () => {
    document.getElementById("startButton").style.display = "none";
    document.getElementById("gameUI").style.display = "block";

    await loadTrainingData();
    await createPoseLandmarker(); // will start the webcam
});

/********************************************************************
 // LOAD TRAINING DATA FROM JSON
 ********************************************************************/
async function loadTrainingData() {
    const response = await fetch("/src/data/handpose-data.json");
    const data = await response.json();

    ["Y", "M", "C", "A"].forEach(label => {
        data[label].forEach(entry => {
            machine.learn(entry.points, label);
        });
    });

    console.log("Training data loaded.");
}

/********************************************************************
 // CREATE THE POSE LANDMARKER
 ********************************************************************/
async function createPoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });

    console.log("Pose model loaded.");
    startWebcam();
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.addEventListener("loadeddata", () => {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        document.querySelector(".videoView").style.height = video.videoHeight + "px";
        webcamRunning = true;
        requestAnimationFrame(predictWebcam);
        cyclePose(); // Start the pose game!
    });
}

/********************************************************************
 // GAME LOGIC
 ********************************************************************/
function cyclePose() {
    currentPose = poseOptions[Math.floor(Math.random() * poseOptions.length)];
    posePrompt.textContent = `Strike the '${currentPose}' pose!`;
    lastPoseChangeTime = performance.now();
    feedback.textContent = "";

    // Start countdown
    let timeLeft = 5;
    timerElement.textContent = timeLeft;
    clearInterval(timerCountdown);

    timerCountdown = setInterval(() => {
        timeLeft--;
        timerElement.textContent = timeLeft;

        if (timeLeft <= 0) {
            clearInterval(timerCountdown);
        }
    }, 1000);
}

/********************************************************************
 // PREDICT + CHECK POSE
 ********************************************************************/
async function predictWebcam() {
    results = await poseLandmarker.detectForVideo(video, performance.now());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks?.length) {
        for (let pose of results.landmarks) {
            drawUtils.drawLandmarks(pose, { radius: 4, color: "#FF0000" });
            drawUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#00FF00" });

            const flatArray = pose.flatMap(p => [p.x, p.y, p.z]);
            const predicted = machine.classify(flatArray);

            const rightWrist = pose[16];
            if (rightWrist) {
                image.style.transform = `translate(${rightWrist.x * video.videoWidth}px, ${rightWrist.y * video.videoHeight}px)`;
            }

            const now = performance.now();
            if (now - lastPoseChangeTime < poseChangeInterval - 1000) {
                if (predicted === currentPose) {
                    feedback.textContent = `✅ Correct pose: ${predicted}`;
                    // Show checkmark
                    checkmark.classList.add("show");

                    // Hide after 1 second
                    setTimeout(() => {
                        checkmark.classList.remove("show");
                    }, 1000);
                } else {
                    feedback.textContent = `❌ Incorrect pose: ${predicted}`;
                }
            } else if (now - lastPoseChangeTime >= poseChangeInterval) {
                cyclePose();
            }
        }
    }

    if (webcamRunning) {
        requestAnimationFrame(predictWebcam);
    }
}
