app.controller('AnalyzeCtrl', function ($scope, $timeout, AnalyzeService, $http) {
    $scope.selectedTestResult = null;

    // 1️⃣ Load Training Data Tree
    AnalyzeService.testResults().then(function (response) {
        $("#testResults").dxTreeList({
            dataSource: response.data,
            keyExpr: "id",
            parentIdExpr: "parentId",
            columns: [
                { dataField: "name", caption: "Name" },
                { dataField: "created_at", caption: "Created At", dataType: "datetime" },
                { dataField: "modified_at", caption: "Modified At", dataType: "datetime", sortOrder: "desc" }
            ],
            showBorders: true,
            autoExpandAll: true,
            wordWrapEnabled: true,
            selection: {
                mode: "single"
            },
            searchPanel: {
                visible: true,
                width: 250,
                placeholder: "Search..."
            },
            onSelectionChanged: function (e) {
                const rowData = e.component.getSelectedRowsData()[0];
                if (!rowData || rowData.type !== "file") {
                    e.component.clearSelection();
                    $scope.selectedTestResult = null;
                } else {
                    $scope.selectedTestResult = rowData.name;
                }
            },
            onInitialized: function (e) {
            }
        });
    });


    // 3️⃣ Add Training Button
    $timeout(function () {
        $("#analyzeButton").dxButton({
            text: "Analyze Test Data",
            type: "success",
            onClick: function () {
                if (!$scope.selectedTestResult || !$scope.selectedTestResult) {
                    $("#serverStatus").text("⚠️ Please select test result.");
                    return;
                }

                $("#serverStatus").text("⏳ Analyzing data...");

                AnalyzeService.testResultData($scope.selectedTestResult).then(function (response) {
                    raw = response.data || { metrics: null, trades: [], used_data: [] };

                    // Parse/derive
                    candles = mapCandles(raw.used_data);
                    const mapped = mapTrades(raw.trades);
                    entries = mapped.entries.filter(p => p.time && p.price != null);
                    exits = mapped.exits.filter(p => p.time); // price may be null if unknown


                    buildChartOptions(candles, entries, exits);



                    $("#serverStatus").text("✅ Got data!");
                })
                    .catch(function () {
                        $("#serverStatus").text("❌ Error.");
                    });
            }
        });
    }, 0);



    // UI state
    $scope.folder = '';             // set a default if you want
    $scope.loading = false;
    $scope.error = '';
    $scope.showEntries = true;
    $scope.showExits = true;

    // Raw data holders
    let raw = { metrics: null, trades: [], used_data: [] };

    // Parsed/derived series
    let candles = [];
    let entryPoints = [];
    let exitPoints = [];


    function num(v) { const n = Number(v); return isFinite(n) ? n : null; }
    function toDate(v) { return v ? new Date(v) : null; }

    function mapCandles(arr) {
        return (arr || []).map(d => {
            const time = toDate(d.time || d.Time || d.timestamp || d.Date);
            const open = num(d.open ?? d.Open);
            const high = num(d.high ?? d.High);
            const low = num(d.low ?? d.Low);
            const close = num(d.close ?? d.Close);
            return (time && open != null && high != null && low != null && close != null)
                ? { time, open, high, low, close }
                : null;
        }).filter(Boolean);
    }

    function mapTrades(arr) {
        return (arr || []).reduce((acc, t) => {
            // entry (optional)
            if (t.entry_time) {
                acc.entries.push({
                    time: toDate(t.entry_time),
                    price: num(t.entry),
                    type: t.type, sl: num(t.sl), tp: num(t.tp), pnl: num(t.pnl), balance: num(t.balance)
                });
            }
            // exit (timestamp is required)
            acc.exits.push({
                time: toDate(t.timestamp),
                price: num(t.exit_price ?? t.tp ?? t.sl),
                type: t.type, sl: num(t.sl), tp: num(t.tp), pnl: num(t.pnl), balance: num(t.balance)
            });
            return acc;
        }, { entries: [], exits: [] });
    }

    // ---- Chart binding ----
    function buildChartOptions(candles, entryPoints, exitPoints) {
        // Base series
        const series = [{
            name: 'Price',
            type: 'candlestick',
            openValueField: 'open',
            highValueField: 'high',
            lowValueField: 'low',
            closeValueField: 'close',
            aggregation: { enabled: true }
        }];

        if ($scope.showEntries && entryPoints.length) {
            series.push({
                name: 'Entries',
                type: 'scatter',
                dataSource: entryPoints,
                argumentField: 'time',
                valueField: 'price',
                point: { size: 10, symbol: 'triangleUp', hoverMode: 'onlyPoint' },
                hoverStyle: { size: 12 }
            });
        }

        if ($scope.showExits && exitPoints.length) {
            series.push({
                name: 'Exits',
                type: 'scatter',
                dataSource: exitPoints,
                argumentField: 'time',
                valueField: 'price',
                point: { size: 10, symbol: 'triangleDown', hoverMode: 'onlyPoint' },
                hoverStyle: { size: 12 }
            });
        }

        const options = {
            // (axes, zoom, etc.)
            series: [
                {
                    name: 'Price',
                    type: 'candlestick',
                    dataSource: candles,          // <— explicit
                    argumentField: 'time',
                    openValueField: 'open',
                    highValueField: 'high',
                    lowValueField: 'low',
                    closeValueField: 'close',
                    aggregation: { enabled: true }
                },
                $scope.showEntries && entries.length ? {
                    name: 'Entries', type: 'scatter',
                    dataSource: entries, argumentField: 'time', valueField: 'price',
                    point: { size: 10, symbol: 'triangleUp' }
                } : null,
                $scope.showExits && exits.length ? {
                    name: 'Exits', type: 'scatter',
                    dataSource: exits, argumentField: 'time', valueField: 'price',
                    point: { size: 10, symbol: 'triangleDown' }
                } : null
            ].filter(Boolean),
        };

        $('#priceChart').dxChart(options).dxChart('instance');

    }

    $scope.rebuildSeries = function () {
        buildChartOptions();
    };
});
