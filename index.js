const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const cors = require('cors');

const { Canvas, Image, ImageData, loadImage } = canvas;

// Registrar canvas como el entorno de faceapi
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());

// Multer para manejar la subida de archivos
const upload = multer({ dest: 'uploads/' });

// Cargar modelos de faceapi.js
const loadModels = async () => {
    const modelPath = path.join(__dirname, 'models');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
    await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
};

loadModels();

app.get('/status', (req, res) => {
    res.send('funcionando');
});

// Ruta para procesar análisis
app.post('/analyze', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        // Cargar la imagen en un objeto canvas compatible
        const img = await loadImage(imagePath);
        const imgCanvas = new Canvas(img.width, img.height);
        const ctx = imgCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, img.width, img.height);

        // Procesar la imagen con face-api.js
        const detections = await faceapi
            .detectSingleFace(imgCanvas)
            .withFaceExpressions()
            .withAgeAndGender();

        // Filtrar solo las expressions y age
        const analysis = {
            expressions: detections.expressions,
            age: detections.age
        };

        // Enviar los resultados al cliente
        res.json(analysis);
    } catch (error) {
        res.status(500).json({ error: 'Error procesando la imagen', details: error.message });
    } finally {
        // Eliminar el archivo temporal después de procesar
        fs.unlinkSync(imagePath);
    }
});


// Ruta para procesar edad
app.post('/analyze/age', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        // Cargar la imagen en un objeto canvas compatible
        const img = await loadImage(imagePath);
        const imgCanvas = new Canvas(img.width, img.height);
        const ctx = imgCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, img.width, img.height);

        // Procesar la imagen con face-api.js
        const detections = await faceapi
            .detectSingleFace(imgCanvas)
            .withAgeAndGender();

        // Filtrar solo la age
        const analysis = {
            age: detections.age
        };

        // Enviar los resultados al cliente
        res.json(analysis);
    } catch (error) {
        res.status(500).json({ error: 'Error procesando la imagen', details: error.message });
    } finally {
        // Eliminar el archivo temporal después de procesar
        fs.unlinkSync(imagePath);
    }
});

// Iniciar el servidor
const PORT = process.env.PORT || 3000
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
})
