const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const PORT = 3000;
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });
app.set('view engine', 'ejs');

const dogBreedTranslations = {
    "Halden Hound dog": "كلب هالدن هاوند", "Hanover Hound dog": "كلب هانوفر هاوند", "Harrier dog": "كلب هارير",
    "Havanese dog": "هافانيز", "Hokkaido dog": "هوكايدو", "Hovawart dog": "هوفاوارت",
    "Hygenhund dog": "هيجنهوند", "Ibizan Hound dog": "كلب الصيد الإيبيزي", "Icelandic Sheep dog": "كلب الرعي الآيسلندي",
    "Irish Red and White Setter dog": "سيتر أيرلندي أحمر وأبيض", "Irish Setter dog": "سيتر أيرلندي", "Irish Terrier dog": "ترير أيرلندي",
    "Irish Water Spaniel dog": "سبانيل الماء الأيرلندي", "Irish Wolf Hound dog": "الكلب الذئبي الإيرلندي", "Istrian Coarse Haired Hound dog": "كلب الصيد الإستري خشن الشعر",
    "Istrian Short Haired Hound dog": "كلب الصيد الإستري قصير الشعر", "Italian Segugio Short Haired Dog": "سيجوجيو إيطالي قصير الشعر", "Jack Russell Terrier dog": "جاك راسل ترير",
    "Jagdterrier dog": "جاغد ترير", "Jamthund dog": "جامتهوند", "Japanese Chin dog": "تشين ياباني",
    "Japanese Spitz dog": "سبيتز ياباني", "Japanese Terrier dog": "ترير ياباني", "Kai Ken dog": "كاي كين",
    "Kangal Shepherd dog": "كلب الراعي كانغال", "Karelian Bear dog": "كلب الدببة الكاريلي", "Karst Shepherd dog": "كلب الراعي كارست",
    "Kerry Blue Terrier dog": "كيري بلو ترير", "King Charles Spaniel dog": "كينغ تشارلز سبانيل", "Kishu dog": "كيشو",
    "Komondor dog": "كوموندور", "Kooikerhondje dog": "كويكرهوندجي", "Korean Jindo dog": "جندو كوري",
    "Kromfohrlander dog": "كرومفورلاندر", "Kuvasz dog": "كوفاز", "Labradoodle dog": "لابرادودل",
    "Labrador Retriever dog": "لابرادور ريتريفر", "Lagotto Romagnolo dog": "لاغوتو روماجنولو", "Lakeland Terrier dog": "ليكلاند ترير",
    "Lapponian herder dog": "راعي لابي", "Large Munsterlander dog": "مونسترلاندر كبير", "Lhasa Apso dog": "لاسا أبسو",
    "Little lion dog": "كلب الأسد الصغير", "Majorca Mastiff dog": "ماستيف ماجوركا", "Majorca Shepherd dog": "راعي ماجوركا",
    "Manchester Terrier dog": "مانشستر ترير", "Maremma and Abruzzes Sheep dog": "كلب الراعي ماريما وأبروز", "Miniature Pinscher dog": "بينشر مصغر",
    "Miniature Schnauzer dog": "شناوزر مصغر", "Montenegrin Mountain Hound dog": "كلب جبال مونتينيغرو", "Mudi dog": "مودي",
    "Norfolk Terrier dog": "نورفولك ترير", "Norrbottenspets dog": "نوربوتن سبيتس", "Norwegian Buhund dog": "بوهوند نرويجي",
    "Norwegian Elkhound dog": "إلكهاوند نرويجي", "Norwegian Lundehund dog": "لوندهوند نرويجي", "Norwich Terrier dog": "نورويتش ترير",
    "Nova Scotia Duck Tolling Retriever dog": "مسترد نوفا سكوشا تولينج", "Dog": "كلب", "Not_Dog": "ليس كلبًا"
};

let verificationModel, breedModel, verificationLabels, breedLabels;
async function loadModels() {
    console.log("Loading models... This might take a while.");
    const verModelUrl = "https://teachablemachine.withgoogle.com/models/8TuehYXuV/";
    const breedModelUrl = "https://teachablemachine.withgoogle.com/models/k89hG5ujt/";

    [verificationModel, breedModel] = await Promise.all([
        tf.loadLayersModel(verModelUrl + 'model.json'),
        tf.loadLayersModel(breedModelUrl + 'model.json')
    ]);

    const [verMetadata, breedMetadata] = await Promise.all([
        fetch(verModelUrl + 'metadata.json').then(res => res.json()),
        fetch(breedModelUrl + 'metadata.json').then(res => res.json())
    ]);

    verificationLabels = verMetadata.labels;
    breedLabels = breedMetadata.labels;
    console.log("Models loaded successfully!");
}

function processImage(buffer) {
    return tf.tidy(() => {
        const tensor = tf.node.decodeImage(buffer, 3);
        const resized = tf.image.resizeBilinear(tensor, [224, 224]);
        const normalized = resized.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
        return normalized.expandDims(0);
    });
}

app.get('/', (req, res) => {
    res.render('index', { result: null, error: null, translations: dogBreedTranslations });
});

app.post('/upload', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.render('index', { result: null, error: "الرجاء رفع ملف صورة.", translations: dogBreedTranslations });
    }
    try {
        const imageBuffer = req.file.buffer;
        const processedImage = processImage(imageBuffer);

        const verPredictionTensor = verificationModel.predict(processedImage);
        const verPredictionData = await verPredictionTensor.data();
        const verIndex = verPredictionData.indexOf(Math.max(...verPredictionData));
        if (verificationLabels[verIndex] !== 'Dog') {
            tf.dispose([processedImage, verPredictionTensor]);
            return res.render('index', { result: null, error: "هذه لا تبدو صورة كلب. الرجاء تجربة صورة أخرى!", translations: dogBreedTranslations });
        }

        const breedPredictionTensor = breedModel.predict(processedImage);
        const breedPredictionData = await breedPredictionTensor.data();
        const predictions = Array.from(breedPredictionData)
            .map((score, i) => ({ class: breedLabels[i], score: score }))
            .sort((a, b) => b.score - a.score);

        const uploadedImage_dataUrl = `data:${req.file.mimetype};base64,${imageBuffer.toString('base64')}`;

        tf.dispose([processedImage, verPredictionTensor, breedPredictionTensor]);

        res.render('index', { result: { predictions, uploadedImage: uploadedImage_dataUrl }, error: null, translations: dogBreedTranslations });
    } catch (e) {
        console.error(e);
        res.render('index', { result: null, error: "حدث خطأ غير متوقع أثناء التحليل.", translations: dogBreedTranslations });
    }
});

loadModels().then(() => {
    app.listen(PORT, () => {
        console.log(`✅ Server is running on http://localhost:${PORT}`);
    });
}).catch(err => {
    console.error("❌ Failed to load models. Check the link in the code. Error:", err);
});