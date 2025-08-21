app.service("TrainingService", function ($http) {
    const baseURL = "http://localhost:8000/api";

    this.getAlgorithms = function () {
        return $http.get(`${baseURL}/algorithms`);
    };


    this.generateAgent = function (selectedTrainingFile, selectedAlgorithmFile) {
        return $http.post(`${baseURL}/agent/train`, {
            dataFile: selectedTrainingFile,
            strategyFile: selectedAlgorithmFile
        });
    };

});
