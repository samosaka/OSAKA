app.service("AnalyzeService", function ($http) {
    const baseURL = "http://localhost:8000/api";

    this.testResults = function () {
        return $http.get(`${baseURL}/analyze/testResults`);
    };

    this.testResultData = function (testResult) {
        return $http.post(`${baseURL}/analyze/testResultData`, {
            testResult: testResult
        });
    };

});
