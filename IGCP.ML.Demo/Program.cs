using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IGCP.ML.Demo
{
    // ---------- ENTRENAMIENTO ----------
    public class TrainRow
    {
        [LoadColumn(0)] public string student_id { get; set; } = "";
        [LoadColumn(1)] public string course_id  { get; set; } = "";
        [LoadColumn(2)] public string term       { get; set; } = "";

        [LoadColumn(3)] public float credits { get; set; }
        [LoadColumn(4)] public float prereq_count { get; set; }

        [LoadColumn(5)] public bool  is_morning { get; set; }
        [LoadColumn(6)] public bool  is_online  { get; set; }

        [LoadColumn(7)] public float lag_course_demand_t1 { get; set; }
        [LoadColumn(8)] public float lag_student_load_t1  { get; set; }

        [LoadColumn(9)] public bool  label { get; set; }   // 1 si se matriculó, 0 si no
    }

    // ---------- PREDICCIÓN ----------
    public class CandidateRow
    {
        [LoadColumn(0)] public string student_id { get; set; } = "";
        [LoadColumn(1)] public string course_id  { get; set; } = "";
        [LoadColumn(2)] public string term       { get; set; } = "";

        [LoadColumn(3)] public float credits { get; set; }
        [LoadColumn(4)] public float prereq_count { get; set; }

        [LoadColumn(5)] public bool  is_morning { get; set; }
        [LoadColumn(6)] public bool  is_online  { get; set; }

        [LoadColumn(7)] public float lag_course_demand_t1 { get; set; }
        [LoadColumn(8)] public float lag_student_load_t1  { get; set; }
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")] public bool Predicted { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    class Program
    {
        static void Main()
        {
            var ml = new MLContext(seed: 7);

            // ---------- 1) Cargar datos de entrenamiento ----------
            Console.WriteLine("Cargando datos de entrenamiento...");
            var trainData = ml.Data.LoadFromTextFile<TrainRow>(
                path: "data/train.csv", hasHeader: true, separatorChar: ',');

            // ---------- 2) Pipeline con LightGBM ----------
            var pipeline =
                ml.Transforms.Categorical.OneHotHashEncoding(new[]
                {
                    // (outputCol, inputCol)
                    new InputOutputColumnPair("course_id_enc", "course_id"),
                    new InputOutputColumnPair("term_enc", "term")
                })
                .Append(ml.Transforms.Conversion.ConvertType(
                    new[]
                    {
                        new InputOutputColumnPair("is_morning", "is_morning"),
                        new InputOutputColumnPair("is_online", "is_online")
                    }, outputKind: DataKind.Single))
                .Append(ml.Transforms.Concatenate("Features",
                    "course_id_enc","term_enc","credits","prereq_count",
                    "is_morning","is_online","lag_course_demand_t1","lag_student_load_t1"))
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.LightGbm(labelColumnName: "label"));

            // ---------- 3) Train/Test split ESTRATIFICADO (SIN cross-validation) ----------
            var allRows = ml.Data.CreateEnumerable<TrainRow>(trainData, reuseRowObject: false).ToList();
            var pos = allRows.Where(r => r.label).ToList();
            var neg = allRows.Where(r => !r.label).ToList();

            if (pos.Count == 0 || neg.Count == 0)
            {
                Console.WriteLine("[DATOS] Se requiere al menos una fila con label=1 y otra con label=0 en todo el dataset.");
                return;
            }

            var testFraction = 0.30;
            var seed = 7;
            var rnd = new Random(seed);

            (List<T> train, List<T> test) Split<T>(List<T> items)
            {
                var shuffled = items.OrderBy(_ => rnd.Next()).ToList();
                int testCount = Math.Max(1, (int)Math.Round(shuffled.Count * testFraction));
                if (testCount >= shuffled.Count) testCount = shuffled.Count - 1; // deja algo para train
                var testPart  = shuffled.Take(testCount).ToList();
                var trainPart = shuffled.Skip(testCount).ToList();
                if (trainPart.Count == 0 && testPart.Count > 1) { trainPart.Add(testPart[0]); testPart.RemoveAt(0); }
                if (testPart.Count == 0 && trainPart.Count > 1) { testPart.Add(trainPart[0]); trainPart.RemoveAt(0); }
                return (trainPart, testPart);
            }

            var (posTrain, posTest) = Split(pos);
            var (negTrain, negTest) = Split(neg);

            var trainList = new List<TrainRow>(); trainList.AddRange(posTrain); trainList.AddRange(negTrain);
            var testList  = new List<TrainRow>(); testList.AddRange(posTest);  testList.AddRange(negTest);

            var trainSet = ml.Data.LoadFromEnumerable(trainList);
            var testSet  = ml.Data.LoadFromEnumerable(testList);

            Console.WriteLine($"[Split] Train={trainList.Count}  Test={testList.Count}  (posTrain={posTrain.Count}, negTrain={negTrain.Count}, posTest={posTest.Count}, negTest={negTest.Count})");

            // ---------- 4) Entrenar y evaluar ----------
            Console.WriteLine("Entrenando modelo (LightGBM)...");
            var model = pipeline.Fit(trainSet);

            var scored = model.Transform(testSet);

            var testHasPos = testList.Any(x => x.label);
            var testHasNeg = testList.Any(x => !x.label);

            if (testHasPos && testHasNeg)
            {
                var metrics = ml.BinaryClassification.Evaluate(scored, labelColumnName: "label");
                Console.WriteLine($"AUC-PR={metrics.AreaUnderPrecisionRecallCurve:F3}  F1={metrics.F1Score:F3}  Acc={metrics.Accuracy:P2}");
            }
            else
            {
                Console.WriteLine("[AVISO] El set de TEST quedó con una sola clase; omito métricas AUC/F1 para evitar error.");
            }

            // Guardar modelo para evidencia
            Directory.CreateDirectory("models");
            using (var fs = File.Create("models/enroll_model.zip"))
                ml.Model.Save(model, trainSet.Schema, fs);

            // ---------- 5) (Demo) Probar conexión a tu BD LocalDB ----------
            try
            {
                // Puedes SOBREESCRIBIR esta cadena con la variable de entorno ML_DEMO_CONNSTR
                var connectionString = @"Server=(localdb)\VerbTables;Database=VerbTables;Trusted_Connection=True;TrustServerCertificate=True;";


                using var conn = new SqlConnection(connectionString);
                conn.Open();

                // Muestra la BD actual e instancia
                using var cmd = new SqlCommand("SELECT DB_NAME();", conn);
                var dbName = (string)cmd.ExecuteScalar();
                Console.WriteLine($"[BD OK] Conectado a: {dbName}  (Servidor: {conn.DataSource})");
            }
            catch (Exception ex)
            {
                Console.WriteLine("[BD ERROR] No se pudo conectar: " + ex.Message);
            }

            // ---------- 6) Predicciones sobre candidatos ----------
            Console.WriteLine("Cargando candidatos del próximo período...");
            var candidates = ml.Data.LoadFromTextFile<CandidateRow>(
                path: "data/candidates_next_term.csv", hasHeader: true, separatorChar: ',');

            var predEngine = ml.Model.CreatePredictionEngine<CandidateRow, Prediction>(model);

            var predicted = ml.Data.CreateEnumerable<CandidateRow>(candidates, reuseRowObject: false)
                .Select(r =>
                {
                    var p = predEngine.Predict(r);
                    return new
                    {
                        r.student_id, r.course_id, r.term,
                        probability = p.Probability
                    };
                })
                .ToList();

            // ---------- 7) Exportar salidas ----------
            // a) Por estudiante-curso
            WriteCsv("predictions_by_student.csv",
                "student_id,course_id,term,probability",
                predicted.Select(x => $"{x.student_id},{x.course_id},{x.term},{x.probability.ToString(CultureInfo.InvariantCulture)}"));

            // b) Demanda agregada por curso
            var demand = predicted
                .GroupBy(x => (x.course_id, x.term))
                .Select(g => new
                {
                    g.Key.course_id,
                    g.Key.term,
                    expected_enrollment = g.Sum(v => v.probability)
                })
                .OrderByDescending(x => x.expected_enrollment)
                .ToList();

            WriteCsv("demand_by_course.csv",
                "course_id,term,expected_enrollment",
                demand.Select(x => $"{x.course_id},{x.term},{x.expected_enrollment.ToString("F2", CultureInfo.InvariantCulture)}"));

            Console.WriteLine("Listo ✅  Salidas: predictions_by_student.csv y demand_by_course.csv");
        }

        static void WriteCsv(string path, string header, IEnumerable<string> rows)
        {
            File.WriteAllLines(path, new[] { header }.Concat(rows));
            Console.WriteLine($"Archivo generado: {path}");
        }
    }
}
