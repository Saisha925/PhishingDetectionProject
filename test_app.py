import unittest
from app import create_app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_index(self):
        with self.app.test_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)

    def test_check_phishing(self):
        with self.app.test_client() as client:
            response = client.post('/check-phishing', data={'emailContent': 'Test email content'})
            self.assertEqual(response.status_code, 200)
            json_data = response.get_json()
            self.assertIn('result', json_data)
            self.assertIn('confidence', json_data)
