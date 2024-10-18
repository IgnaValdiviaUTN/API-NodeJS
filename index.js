const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs');
const sharp = require('sharp');
const cors = require('cors');

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
        // Usar sharp para cargar y convertir la imagen a un formato que se pueda manipular
        const { data, info } = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });

        // Convertir la imagen a un tensor de 3 dimensiones [ancho, alto, canales (RGB)]
        const imgTensor = tf.tensor3d(data, [info.height, info.width, 3]);

        // Procesar la imagen con face-api.js
        const detections = await faceapi
            .detectSingleFace(imgTensor)
            .withFaceExpressions()
            .withAgeAndGender();

        // Liberar el tensor
        imgTensor.dispose();

        // Filtrar solo las expressions y age
        const analysis = {
            expressions: detections.expressions,
            age: detections.age
        };

        // Enviar los resultados al cliente
        res.json(analysis);
    } catch (error) {
        console.error('Error procesando la imagen:', error);
        res.status(500).send('Error procesando la imagen');
    } finally {
        // Eliminar el archivo temporal después de procesar
        fs.unlinkSync(imagePath);
    }
});

// Ruta para procesar edad
app.post('/analyze/age', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {

        const { data, info } = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });
        const imgTensor = tf.tensor3d(data, [info.height, info.width, 3]);

        // Procesar la imagen con face-api.js
        const detections = await faceapi
            .detectSingleFace(imgTensor)
            .withAgeAndGender();

        imgTensor.dispose();

        // Filtrar solo la edad
        const analysis = {
            age: detections.age
        };

        // Enviar los resultados al cliente
        res.json(analysis);
    } catch (error) {
        res.status(500).send('Error procesando la imagen');
    } finally {
        // Eliminar el archivo temporal después de procesar
        fs.unlinkSync(imagePath);
    }
});


// Iniciar el servidor
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});



