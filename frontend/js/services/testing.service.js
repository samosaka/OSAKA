app.service("TestingService", function ($http) {
    const baseURL = "http://localhost:8000/api";

    this.getAgents = function () {
        return $http.get(`${baseURL}/agent`);
    };


    this.getBacktests = function () {
        return $http.get(`${baseURL}/backtest`);
    };
    

    this.startTest = function (selectedTestingFile, selectedAgent, backtestFile) {
        return $http.post(`${baseURL}/agent/test`, {
            dataFile: selectedTestingFile,
            agentFile: selectedAgent,
            backtestFile: backtestFile
        });
    };

});
