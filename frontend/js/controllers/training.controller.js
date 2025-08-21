app.controller('TrainingCtrl', function ($scope, $timeout, HistoryService, TrainingService, $http) {
    $scope.selectedTrainingFile = null;
    $scope.selectedAlgorithmFile = null;

    // 1️⃣ Load Training Data Tree
    HistoryService.getHistoricalTree().then(function (response) {
        $("#trainingData").dxTreeList({
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
                    $scope.selectedTrainingFile = null;
                } else {
                    $scope.selectedTrainingFile = rowData.name;
                }
            },
            onInitialized: function (e) {
                $scope.trainingDataTreeInstance = e.component;
            }
        });
    });

    // 2️⃣ Load Algorithm Tree
    function loadAlgorithmsList() {
        TrainingService.getAlgorithms().then(function (response) {
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
                        $scope.selectedAlgorithmFile = null;
                    } else {
                        $scope.selectedAlgorithmFile = rowData.name;
                    }
                },
                onInitialized: function (e) {
                    $scope.trainingAlgorithmTreeInstance = e.component;
                }
            };

            if ($scope.trainingAlgorithmTreeInstance) {
                $scope.trainingAlgorithmTreeInstance.option("dataSource", response.data);
            } else {
                $("#trainingAlgorithm").dxTreeList(config);
            }
        });
    }

    loadAlgorithmsList();

    // 3️⃣ Add Training Button
    $timeout(function () {
        $("#startTrainingButton").dxButton({
            text: "🚀 Generate (Train) Agent",
            type: "success",
            onClick: function () {
                if (!$scope.selectedTrainingFile || !$scope.selectedAlgorithmFile) {
                    $("#trainingStatus").text("⚠️ Please select both a training data file and a algorithm file.");
                    return;
                }

                $("#trainingStatus").text("⏳ Starting training...");

                TrainingService.generateAgent($scope.selectedTrainingFile, $scope.selectedAlgorithmFile).then(function (response) {
                    $("#trainingStatus").text("✅ Training started successfully!");
                })
                    .catch(function () {
                        $("#trainingStatus").text("❌ Error starting training.");
                    });
            }
        });
    }, 0);
});
