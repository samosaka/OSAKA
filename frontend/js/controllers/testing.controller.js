app.controller('TestingCtrl', function ($scope, $timeout, HistoryService, TestingService) {
    $scope.selectedAgentFile = null;
    $scope.selectedAgentFile = null;

    // 1️⃣ Load Agent Tree
    HistoryService.getHistoricalTree().then(function (response) {
        $("#testingData").dxTreeList({
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
                    $scope.selectedTestingFile = null;
                } else {
                    $scope.selectedTestingFile = rowData.name;
                }
            },
            onInitialized: function (e) {
                $scope.testingDataTreeInstance = e.component;
            }
        });
    });

    // 2️⃣ Load Algorithm Tree
    function loadAgentList() {
        TestingService.getAgents().then(function (response) {
            const config = {
                dataSource: response.data,
                keyExpr: "id",
                parentIdExpr: "parentId",
                columns: [
                    { dataField: "name", caption: "Name" },
                    { dataField: "created_at", caption: "Created At", dataType: "datetime" },
                    { dataField: "modified_at", caption: "Modified At", dataType: "datetime", sortOrder: "desc" }
                ],
                selection: { mode: "single" },
                searchPanel: { visible: true, width: 250 },
                showBorders: true,
                autoExpandAll: true,
                wordWrapEnabled: true,
                onSelectionChanged: function (e) {
                    const rowData = e.component.getSelectedRowsData()[0];
                    if (!rowData || rowData.type !== "file") {
                        e.component.clearSelection();
                        $scope.selectedAgentFile = null;
                    } else {
                        $scope.selectedAgentFile = rowData.parentId + '\\' + rowData.name;
                    }
                },
                onInitialized: function (e) {
                    $scope.agentTreeInstance = e.component;
                }
            };

            if ($scope.agentTreeInstance) {
                $scope.agentTreeInstance.option("dataSource", response.data);
            } else {
                $("#testingAgent").dxTreeList(config);
            }
        });
    }


    // 2️⃣ Load backtest Tree
    function loadBacktestList() {
        TestingService.getBacktests().then(function (response) {
            const config = {
                dataSource: response.data,
                keyExpr: "id",
                parentIdExpr: "parentId",
                columns: [
                    { dataField: "name", caption: "Name" },
                    { dataField: "created_at", caption: "Created At", dataType: "datetime" },
                    { dataField: "modified_at", caption: "Modified At", dataType: "datetime", sortOrder: "desc" }
                ],
                selection: { mode: "single" },
                searchPanel: { visible: true, width: 250 },
                showBorders: true,
                autoExpandAll: true,
                wordWrapEnabled: true,
                onSelectionChanged: function (e) {
                    const rowData = e.component.getSelectedRowsData()[0];
                    if (!rowData || rowData.type !== "file") {
                        e.component.clearSelection();
                        $scope.selectedBacktestFile = null;
                    } else {
                        $scope.selectedBacktestFile = rowData.name;
                    }
                },
                onInitialized: function (e) {
                    $scope.backtestTreeInstance = e.component;
                }
            };

            if ($scope.backtestTreeInstance) {
                $scope.backtestTreeInstance.option("dataSource", response.data);
            } else {
                $("#testingBacktest").dxTreeList(config);
            }
        });
    }
    loadAgentList();
    loadBacktestList();

    // 3️⃣ Add Testing Button
    $timeout(function () {
        $("#startTestingButton").dxButton({
            text: "🚀 Test Agent",
            type: "success",
            onClick: function () {
                if (!$scope.selectedTestingFile || !$scope.selectedAgentFile) {
                    $("#testingStatus").text("⚠️ Please select both a testing data file and a agent file.");
                    return;
                }

                $("#testingStatus").text("⏳ Starting testing...");

                TestingService.startTest($scope.selectedTestingFile, $scope.selectedAgentFile, $scope.selectedBacktestFile).then(function (response) {
                    $("#testingStatus").text("✅ Testing started successfully!");
                })
                    .catch(function () {
                        $("#testingStatus").text("❌ Error starting testing.");
                    });
            }
        });
    }, 0);
});
