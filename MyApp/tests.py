from django.test import TestCase


class SmokeTests(TestCase):
    def test_homepage(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

