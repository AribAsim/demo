import requests
import time
import json
import sys
import random
import string
from typing import List, Dict, Any, Optional

# --- Configuration ---
BASE_URL = "http://localhost:8000/api"
EMAIL = f"tester_{int(time.time())}@example.com" # Unique email each run
PASSWORD = "TestPassword123!"
NAME = "Stress Tester"
CHILD_NAME = "Test Child"
CHILD_AGE = 10

# --- ANSI Colors ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Back = type('Back', (object,), {'GREEN': '\033[42m'}) # Adding Back for background
    Black = '\033[30m' # Adding black text
    BAIL = '\033[91m' # Fallback for BAIL if not present

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*10} {msg} {'='*10}{Colors.ENDC}")

def print_pass(msg):
    print(f"{Colors.OKGREEN}[PASS] {msg}{Colors.ENDC}")

def print_fail(msg):
    print(f"{Colors.FAIL}[FAIL] {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}[INFO] {msg}{Colors.ENDC}")

class SafeBrowseStressTest:
    def __init__(self):
        self.token = None
        self.headers = {}
        self.profile_id = None
        self.user_id = None
        self.results = {"pass": 0, "fail": 0}

    def run_all(self):
        print_header("SAFEBROWSE END-TO-END STRESS TEST")
        
        try:
            if not self.check_server_health():
                print_fail("Server is not running at http://localhost:8000. Please start the backend.")
                return

            self.setup_auth()
            self.setup_profile()
            
            self.test_backend_robustness()
            self.test_search_regex_logic()
            self.test_search_engine_handling()
            self.test_nsfw_content_analysis()
            self.test_digital_wellbeing_logging()
            
            self.print_summary()
            
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Test interrupted by user.{Colors.ENDC}")
        except Exception as e:
            print_fail(f"Critical Test Suite Error: {e}")
            import traceback
            traceback.print_exc()

    def check_server_health(self):
        try:
            requests.get("http://localhost:8000/docs", timeout=2)
            return True
        except:
            return False

    def setup_auth(self):
        print_header("1. Authentication Setup")
        
        # 1. Signup
        print_info(f"Attempting signup with {EMAIL}...")
        try:
            res = requests.post(f"{BASE_URL}/auth/signup", json={
                "email": EMAIL,
                "password": PASSWORD,
                "name": NAME
            })
            if res.status_code == 200:
                data = res.json()
                self.token = data['access_token']
                self.user_id = data['user']['id']
                print_pass("Signup successful")
            else:
                # Fallback to login if user exists (unlikely given timestamp email)
                print_info("Signup failed (User might exist?), trying login...")
                res = requests.post(f"{BASE_URL}/auth/login", json={
                    "email": EMAIL,
                    "password": PASSWORD
                })
                if res.status_code == 200:
                    data = res.json()
                    self.token = data['access_token']
                    self.user_id = data['user']['id']
                    print_pass("Login successful")
                else:
                    raise Exception(f"Auth failed: {res.text}")
            
            self.headers = {"Authorization": f"Bearer {self.token}"}
            
        except Exception as e:
            print_fail(f"Auth Setup Failed: {e}")
            sys.exit(1)

    def setup_profile(self):
        print_info("Setting up child profile...")
        try:
            # Check existing
            res = requests.get(f"{BASE_URL}/profiles", headers=self.headers)
            profiles = res.json()
            if profiles:
                self.profile_id = profiles[0]['id']
                print_pass(f"Using existing profile: {self.profile_id}")
            else:
                # Create new
                res = requests.post(f"{BASE_URL}/profiles", headers=self.headers, json={
                    "name": CHILD_NAME,
                    "age": CHILD_AGE,
                    "maturity_level": "moderate"
                })
                if res.status_code == 200:
                    self.profile_id = res.json()['id']
                    print_pass(f"Created new profile: {self.profile_id}")
                else:
                    raise Exception(f"Profile creation failed: {res.text}")
        except Exception as e:
            print_fail(f"Profile Setup Failed: {e}")
            sys.exit(1)

    def _assert(self, condition, message):
        if condition:
            print_pass(message)
            self.results["pass"] += 1
        else:
            print_fail(message)
            self.results["fail"] += 1

    def test_backend_robustness(self):
        print_header("2. Backend Robustness Testing")
        
        # Test 1: Empty Body
        print_info("Sending empty body to analysis endpoint...")
        try:
            res = requests.post(f"{BASE_URL}/content/analyze", json={}, headers=self.headers)
            self._assert(res.status_code == 422, "Empty body should return 422 (Validation Error)")
        except Exception as e:
            print_fail(f"Request failed: {e}")

        # Test 2: Missing Fields
        print_info("Sending partial body (missing 'content')...")
        try:
            res = requests.post(f"{BASE_URL}/content/analyze", json={"profile_id": self.profile_id, "content_type": "text"}, headers=self.headers)
            self._assert(res.status_code == 422, "Missing fields should return 422")
        except:
            print_fail("Request failed")
            
        # Test 3: Invalid JSON types
        print_info("Sending invalid type for 'profile_id'...")
        try:
            res = requests.post(f"{BASE_URL}/content/analyze", json={
                "profile_id": 12345, # Should be string
                "content_type": "text",
                "content": "test"
            }, headers=self.headers)
            # Pydantic might coerce int to str, check reaction
            if res.status_code == 200:
                 print_info("Integer profile_id was coerced (Acceptable implementation)")
            else:
                 self._assert(res.status_code == 422, "Invalid types should start validation error")
        except:
            pass

        # Test 4: Extremely Long Content (Stress)
        print_info("Sending 100KB text payload...")
        long_text = "safe " * 20000
        start = time.time()
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "text",
            "content": long_text
        }, headers=self.headers)
        duration = time.time() - start
        self._assert(res.status_code == 200, f"Large payload handled in {duration:.2f}s")
        self._assert(res.json()['is_safe'] == True, "Large safe payload returned safe")

    def test_search_regex_logic(self):
        print_header("3. Search & Regex Stress Testing")
        
        test_cases = [
            # False Positives (Must be SAFE)
            ("lo", True, "Short query 'lo'"),
            ("se", True, "Short query 'se'"),
            ("an", True, "Short query 'an'"),
            ("London", True, "Location 'London'"),
            ("Essex", True, "Location 'Essex'"),
            ("Sussex", True, "Location 'Sussex'"),
            ("sex education", True, "Educational query"),
            ("human anatomy diagram", True, "Educational query"),
            ("biology reproduction", True, "Educational query"),
            ("bhabhi", True, "Cultural term (often false positive)"),
            
            # True Positives (Must be UNSAFE)
            ("porn", False, "Explicit 'porn'"),
            ("xxx videos", False, "Explicit 'xxx'"),
            ("hot sex", False, "Explicit phrase"),
            ("how to kill myself", False, "Self-harm phrase"),
            ("buy cocaine online", False, "Drug phrase"),
            ("online casino hacks", False, "Gambling phrase")
        ]
        
        for text, expected_safe, desc in test_cases:
            res = requests.post(f"{BASE_URL}/content/analyze", json={
                "profile_id": self.profile_id,
                "content_type": "text",
                "content": text
            }, headers=self.headers)
            
            if res.status_code == 200:
                is_safe = res.json()['is_safe']
                self._assert(is_safe == expected_safe, f"{desc}: Expected {'SAFE' if expected_safe else 'UNSAFE'}, Got {'SAFE' if is_safe else 'UNSAFE'}")
                if is_safe != expected_safe:
                    print_info(f"   Reason from server: {res.json()['reasons']}")
            else:
                print_fail(f"{desc}: Server Error {res.status_code}")

    def test_search_engine_handling(self):
        print_header("4. Search Engine Handling")
        
        # Google Search - SAFE
        url_safe = "https://www.google.com/search?q=flowers+and+gardens"
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "url",
            "content": url_safe
        }, headers=self.headers)
        self._assert(res.json()['is_safe'] == True, "Google Search (Flowers) should be SAFE")
        
        # Google Search - UNSAFE
        url_unsafe = "https://www.google.com/search?q=hardcore+porn+videos"
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "url",
            "content": url_unsafe
        }, headers=self.headers)
        self._assert(res.json()['is_safe'] == False, "Google Search (Porn) should be UNSAFE")
        
        # DuckDuckGo Check
        url_ddg = "https://duckduckgo.com/?q=math+homework"
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "url",
            "content": url_ddg
        }, headers=self.headers)
        self._assert(res.json()['is_safe'] == True, "DuckDuckGo (Math) should be SAFE")

    def test_nsfw_content_analysis(self):
        print_header("5. NSFW & Content Blocking")
        
        # Test URL Blacklist logic implicit in analysis
        explicit_domain = "https://www.pornhub.com/view_video.php?viewkey=12345"
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "url",
            "content": explicit_domain
        }, headers=self.headers)
        self._assert(res.json()['is_safe'] == False, "Known Adult Domain should be BLOCKED")

        # Test Text Block
        explicit_text = "This story contains extreme violence and murder and gore."
        res = requests.post(f"{BASE_URL}/content/analyze", json={
            "profile_id": self.profile_id,
            "content_type": "text",
            "content": explicit_text
        }, headers=self.headers)
        self._assert(res.json()['is_safe'] == False, "Violence text should be BLOCKED")

    def test_digital_wellbeing_logging(self):
        print_header("7. Digital Wellbeing Verification")
        
        # Generate some activity first (we already did in previous steps)
        # Check logs endpoint
        print_info("Fetching logs...")
        time.sleep(1) # Let async DB writes settle
        
        try:
            res = requests.get(f"{BASE_URL}/logs?profile_id={self.profile_id}", headers=self.headers)
            if res.status_code == 200:
                logs = res.json()
                self._assert(len(logs) > 0, f"Logs retrieved successfully (Count: {len(logs)})")
                
                # Check for specific logs from our test
                has_blocked = any(not log['is_safe'] for log in logs)
                self._assert(has_blocked, "Found BLOCKED entries in logs")
            else:
                print_fail("Failed to fetch logs")
        except Exception as e:
            print_fail(f"Log fetch error: {e}")

        # Check Wellbeing Stats
        print_info("Fetching wellbeing stats...")
        try:
            res = requests.get(f"{BASE_URL}/parent/digital-wellbeing/{self.profile_id}", headers=self.headers)
            if res.status_code == 200:
                stats = res.json()
                self._assert(stats['unsafe_detections_total'] > 0, "Wellbeing stats reflect unsafe detections")
                print_info(f"   Total Unsafe: {stats['unsafe_detections_total']}")
                print_info(f"   Daily Stats: {len(stats['daily_stats'])} days")
            else:
                print_fail("Failed to fetch wellbeing stats")
        except Exception as e:
            print_fail(f"Wellbeing stats error: {e}")

    def print_summary(self):
        print_header("TEST SUMMARY")
        total = self.results["pass"] + self.results["fail"]
        print(f"Total Tests: {total}")
        print(f"{Colors.OKGREEN}Passed:      {self.results['pass']}{Colors.ENDC}")
        if self.results["fail"] > 0:
            print(f"{Colors.FAIL}Failed:      {self.results['fail']}{Colors.ENDC}")
            print(f"\n{Colors.BAIL}Recommendation: Review the failed cases above in server logs/logic.{Colors.ENDC}")
        else:
            print(f"\n{Colors.Back.GREEN}{Colors.Black} SUCCESS: SYSTEM IS DEMO READY {Colors.ENDC}")

if __name__ == "__main__":
    tester = SafeBrowseStressTest()
    tester.run_all()
