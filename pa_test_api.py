import unittest
import requests


class APITestCase(unittest.TestCase):
    def test_predict_endpoint(self):
        # Données de test
        data = {
            "PAYMENT_RATE": 0.0778341260591552,
            "EXT_SOURCE_2": 0.6793073376533056,
            "EXT_SOURCE_3": 0.5513812618027899,
            "DAYS_BIRTH": -14250,
            "DAYS_REGISTRATION": -8295.0,
            "DAYS_LAST_PHONE_CHANGE": -2142.0,
            "DAYS_ID_PUBLISH": -4722,
            "DAYS_EMPLOYED": -1136.0,
            "DAYS_EMPLOYED_PERC": 0.07971929824561404,
            "AMT_GOODS_PRICE": 238500.0,
            "AMT_ANNUITY": 22239.0,
            "ANNUITY_INCOME_PERC": 0.023533333333333333,
            "AMT_CREDIT": 285723.0,
            "REGION_POPULATION_RELATIVE": 0.032561,
            "INCOME_PER_PERSON": 472500.0
        }

        # Effectuer une requête POST vers l'API
        response = requests.post('http://api-patriciadubray.pythonanywhere.com/predict', json=data)

        # Vérifier le code de statut HTTP
        self.assertEqual(response.status_code, 200)

        # Vérifier la structure de la réponse JSON
        json_data = response.json()
        self.assertIn("class_0", json_data)
        self.assertIn("class_1", json_data)
        self.assertIn("summary_plot_path", json_data)

    def test_download_summary_plot_endpoint(self):
        # Effectuer une requête GET pour télécharger le fichier de tracé résumé
        response = requests.get('http://api-patriciadubray.pythonanywhere.com/summary_plot')

        # Vérifier le code de statut HTTP
        self.assertEqual(response.status_code, 200)

        # Vérifier le type de contenu du fichier
        self.assertEqual(response.headers['Content-Type'], 'image/png')

        # Vérifier que le contenu de la réponse n'est pas vide
        self.assertIsNotNone(response.content)


if __name__ == '__main__':
    unittest.main()