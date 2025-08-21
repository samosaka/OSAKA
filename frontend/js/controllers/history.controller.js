app.controller('HistoryCtrl', function ($scope, $timeout, HistoryService) {
    let selectedSymbol = 'EURUSD';
    let selectedTimeFrame = 'M5';

    const timeframes = [
        { id: "M1", name: "1 Minute" },
        { id: "M5", name: "5 Minutes" },
        { id: "M15", name: "15 Minutes" },
        { id: "M30", name: "30 Minutes" },
        { id: "H1", name: "1 Hour" },
        { id: "H4", name: "4 Hours" },
        { id: "D1", name: "1 Day" },
        { id: "W1", name: "1 Week" },
        { id: "MN1", name: "1 Month" }
    ];

    function loadTreeList() {
        HistoryService.getHistoricalTree().then(function (response) {
            if ($scope.treeInstance) {
                $scope.treeInstance.option("dataSource", response.data)
            } else {
                $("#treeList").dxTreeList({
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
                    onInitialized: function (e) {
                        $scope.treeInstance = e.component;
                    }
                });
            }
        });
    }

    function loadForm() {
        HistoryService.getSymbols().then(function (response) {
            const symbolOptions = response.data;

            $("#historyFormContainer").dxForm({
                formData: {
                    selectedSymbol,
                    selectedTimeFrame
                },
                items: [
                    {
                        itemType: "group",
                        colCount: 3,
                        items: [
                            {
                                dataField: "selectedSymbol",
                                editorType: "dxSelectBox",
                                label: { text: "Symbol" },
                                editorOptions: {
                                    dataSource: symbolOptions,
                                    displayExpr: "name",
                                    valueExpr: "id",
                                    placeholder: "Select a symbol",
                                    searchEnabled: true,
                                    onValueChanged(e) {
                                        selectedSymbol = e.value;
                                    }
                                }
                            },
                            {
                                dataField: "selectedTimeFrame",
                                editorType: "dxSelectBox",
                                label: { text: "Timeframe" },
                                editorOptions: {
                                    dataSource: timeframes,
                                    displayExpr: "name",
                                    valueExpr: "id",
                                    placeholder: "Select Timeframe",
                                    onValueChanged(e) {
                                        selectedTimeFrame = e.value;
                                    }
                                }
                            },
                            {
                                itemType: "button",
                                horizontalAlignment: "left",
                                buttonOptions: {
                                    text: "Get Historical Data",
                                    type: "success",
                                    onClick: function () {
                                        if (!selectedSymbol || !selectedTimeFrame) {
                                            $("#responseMsg").text("⚠️ Please select both symbol and timeframe.");
                                            return;
                                        }

                                        HistoryService.getHistoricalData(selectedSymbol, selectedTimeFrame)
                                            .then((res) => {
                                                loadTreeList(); // refresh after successful call
                                            })
                                            .catch(() => {
                                                $("#responseMsg").text("❌ Error contacting server.");
                                            });
                                    }
                                }
                            }
                        ]
                    }
                ]
            });
        });
    }

    loadForm();
    loadTreeList();
    // Init
});
