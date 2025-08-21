app.service("HistoryService", function($http) {
    const baseURL = "http://localhost:8000/api";

    this.getSymbols = function() {
        return $http.get(`${baseURL}/symbols`);
    };

    this.getHistoricalTree = function() {
        return $http.get(`${baseURL}/getHistoricalTree`);
    };

    this.getHistoricalData = function(symbol, timeframe) {
        return $http.post(`${baseURL}/getHistorical`, {
            selectedSymbol: symbol,
            selectedTimeFrame: timeframe
        });
    };
});
