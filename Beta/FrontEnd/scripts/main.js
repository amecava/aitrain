///////////////////////////////////////////////////////
// Parte di define/variabili globali
allenamento_vuoto = {
    "workload_mean": 0,
    //"perceivedWorkload_mean": 0,
    "metabolicWorkload_mean": 0,
    "mechanicalWorkload_mean": 0,
    "kinematicWorkload_mean": 0
};

indirizzo_base = 'http://localhost:8080/res/';
///////////////////////////////

$(document).ready(function() {
    //Mettere tutte le funzioni da inizializzare all'avvio dell'app
    //LoginCheck();
    RadarReload()
    CreateChart();
    loadPlayer();
});

function LoginCheck() {
    $(document).on("submit", "#formLogin", function() {
        var username = $("#username").val();
        var passwd = $("#passwd").val();

        if (username == "prova" && passwd == "prova") {
            window.location = "dashboard.html";
        }
    });
}

function RadarReload() {
    $(document).on("change", "#option_s3,#option_s8", function() {
        CreateChart();
        loadPlayer();
    });
}

function CreateChart() {
    /*var ctx = document.getElementById('myChart').getContext('2d');
    var chart = new Chart(ctx);

    allenamento1 = CalcolaAllenamento(LeggiEsercizi());
    allenamento_labels = getKeys(allenamento1);
    allenamento_values = getValues(allenamento1);

    chart.clear();
    chart.destroy();
    if (typeof chart != 'undefined') {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        // The type of chart we want to create
        type: 'radar',

        // The data for our dataset

        data: {
            labels: allenamento_labels,
            datasets: [{
                label: "My First dataset",
                fill: 'true',
                backgroundColor: 'rgba(255, 255, 255,0.5)',
                borderColor: 'rgb(255, 255, 255)',
                borderWidth: '3',
                data: allenamento_values,
            }]
        },

        // Configuration options go here
        options: {}
    });*/

    //Amcharts
    allenamento1 = CalcolaAllenamento(LeggiEsercizi());
    allenamento1["workload_mean"] *= 20;
    allenamento_labels = getKeys(allenamento1);
    allenamento_values = getValues(allenamento1);
    allenamento_teorico = getValues(AllenamentoTeorico());

    var chart = AmCharts.makeChart("divchart", {
        "type": "radar",
        "theme": "dark",
        "dataProvider": [{
            "name": allenamento_labels[0].split("_", 1),
            "value": allenamento_values[0],
            "teorico": allenamento_teorico[0]
        }, {
            "name": allenamento_labels[1].split("_", 1),
            "value": allenamento_values[1],
            "teorico": allenamento_teorico[1]
        }, {
            "name": allenamento_labels[2].split("_", 1),
            "value": allenamento_values[2],
            "teorico": allenamento_teorico[2]
        }, {
            "name": allenamento_labels[3].split("_", 1),
            "value": allenamento_values[3],
            "teorico": allenamento_teorico[3]
        }],
        "valueAxes": [{
            "axisTitleOffset": 20,
            "minimum": 0,
            "maximum": 80,
            "axisAlpha": 0.15
        }],
        "graphs": [{
                "balloonText": "[[value]]",
                "bullet": "round",
                "lineThickness": 2,
                "valueField": "value"
            },
            {
                "balloonText": "[[value]]",
                "bullet": "round",
                "lineThickness": 2,
                "valueField": "teorico"
            }
        ],
        "categoryField": "name",
        "export": {
            "enabled": true
        }
    });
}

// funzione che prende un oggetto e restituisce un array con le sue keys
function getKeys(obj) {
    var keys = [];
    for (var key in obj) {
        keys.push(key);
    }
    return keys;
}

// funzione che prende un oggetto e restituisce un array con i suoi valori
function getValues(obj) {
    var values = [];
    for (var key in obj) {
        values.push(obj[key]);
    }
    return values;
}

function loadPlayer() {
    var html = '<div class="box_pl">\
            <div id="card_pl">\
                <div id="blur">\
                    <div id="color"></div>\
                </div>\
                <div id="profile">\
                    <img class="profile_img" src="https://www.isfahanlaw.ir/wp-content/uploads/2018/02/Male-Avatar.jpg" alt="User" align="center"/>\
                    <br>\
                    <br>\
                    <div class="card-body-lp">\
                        <h2 class="name">{0}</h2>\
                        <div class="stats">\
                            <div class="stat_left">\
                            <span>Go Score</span>\
                            <span class="value">{1}</span>\
                            </div>\
                            <div class="stat">\
                            <span>Condition</span>\
                            <span class="value">{2}</span>\
                            </div>\
                            <div class="stat_right">\
                            <span>Trend</span>\
                            <span class="value">{3}</span>\
                            </div>\
                        </div>\
                    </div>\
                </div>\
            </div>\
            </div>';
    var result = 0;
    result = Predizione();
    $(".box_pl").remove();
    for (var i = 0; i < result.length; i++) {
        var thisHtml = 0;
        thisHtml = html;
        thisHtml = thisHtml.replace("{0}", result[i].playerName);
        thisHtml = thisHtml.replace("{1}", result[i]['Go_score+1'].toFixed(1));
        thisHtml = thisHtml.replace("{2}", GoScoreCondition(result[i]['Go_score+1']));
        thisHtml = thisHtml.replace("{3}", GoScoreTrend(result[i]['Go_score+1'], result[i].Go_score));
        $(".playerList").append(thisHtml);
    }

}

function GoScoreCondition(val) {
    if (val > 4.5) { return "<div class='square green'></div>"; }
    if (val > 4 && val <= 4.5) { return "<div class='square yellow'></div>"; }
    return "<div class='square red'></div>";
}

function GoScoreTrend(val1, val2) {
    if (val1 > val2) { return "<div class='square green'></div>"; }
    if (val1 == val2) { return "<div class='square yellow'></div>"; }
    return "<div class='square red'></div>";
}

//funzione che prende un nome di un file di testo contenente un JSON e va a prendere la risorsa
// con quel nome e parsa il json. Restituisce il JSON parsato
function CaricaJSON(name) {
    var xyz = null

    $.ajax({
        url: indirizzo_base + name,
        async: false,
        success: function(data) {
            xyz = JSON.parse(data);
        }
    });
    return (xyz);
}

// prende un array di json e restituisce l'array di splitName (nome esercizio)
function ElencoEsercizi(array_json_esercizi) {
    var elenco = [];

    for (i = 0; i < array_json_esercizi.length; i++) {
        elenco.push(array_json_esercizi[i].splitName);
    }
    return elenco;
}

function AllenamentoTeorico() {
    var sel = document.getElementById('option_s8');

    dati_allenamento_teorico = allenamento_vuoto;

    if (sel.value === "md-1") {

        metabolicWorkload_mean = 35;
        kinematicWorkload_mean = 5;
        mechanicalWorkload_mean = 25;
        workload_mean = 20;

    }
    if (sel.value === "md-2") {

        metabolicWorkload_mean = 25;
        kinematicWorkload_mean = 5;
        mechanicalWorkload_mean = 20;
        workload_mean = 15;

    }
    if (sel.value === "md-3") {

        metabolicWorkload_mean = 55;
        kinematicWorkload_mean = 25;
        mechanicalWorkload_mean = 65;
        workload_mean = 50;

    }
    if (sel.value === "md-4") {

        metabolicWorkload_mean = 65;
        kinematicWorkload_mean = 25;
        mechanicalWorkload_mean = 75;
        workload_mean = 55;

    }
    if (sel.value === "md-5") {

        metabolicWorkload_mean = 45;
        kinematicWorkload_mean = 45;
        mechanicalWorkload_mean = 50;
        workload_mean = 45;

    }

    dati_allenamento_teorico['workload_mean'] = workload_mean;
    dati_allenamento_teorico['metabolicWorkload_mean'] = metabolicWorkload_mean;
    dati_allenamento_teorico['mechanicalWorkload_mean'] = mechanicalWorkload_mean;
    dati_allenamento_teorico['kinematicWorkload_mean'] = kinematicWorkload_mean;

    return dati_allenamento_teorico;


}

// prende in entrata un array di nomi esercizi e calcola la somma di tutti i parametri medi di quell'esercizio
function CalcolaAllenamento(array_esercizi_selezionati) {
    dati_esercizi = CaricaJSON('Esercizi.txt')

    ///////////////////////////////////////////////////////////////////////////////////
    dati_allenamento = {
            "workload_mean": 0,
            //"perceivedWorkload_mean": 0,
            "metabolicWorkload_mean": 0,
            "mechanicalWorkload_mean": 0,
            "kinematicWorkload_mean": 0
        }
        ////////////////////////////////////////////////////////////////////////////////////

    function checkName(json_esercizio) {
        return json_esercizio.splitName === nome_esercizio;
    }

    for (i in array_esercizi_selezionati) {
        nome_esercizio = array_esercizi_selezionati[i]
        j = dati_esercizi.findIndex(checkName)

        var keys = getKeys(dati_allenamento)

        for (l in keys) {
            key = keys[l]
            dati_allenamento[key] += dati_esercizi[j][key]
        }
    }
    return dati_allenamento;
}

// funzione che popola il dropdown di scela con i possibili esercizi contenuti in Esercizi.txt
window.onload = function() {
        var select = document.getElementById("option_s3");
        var options = ElencoEsercizi(CaricaJSON("Esercizi.txt"));
        for (var i = 0; i < options.length; i++) {
            var opt = options[i];
            var el = document.createElement("option");
            el.textContent = opt;
            el.value = opt;
            select.appendChild(el);
        }
    }
    // funzione che legge i valori scelti nel dropdown
function LeggiEsercizi() {
    var InvForm = document.forms.form;
    var SelBranchVal = [];
    var x = 0;
    for (x = 0; x < InvForm.param.length; x++) {
        if (InvForm.param[x].selected) {
            //alert(InvForm.kb[x].value);
            SelBranchVal.push(InvForm.param[x].value);
        }
    }
    return SelBranchVal;
}

//funzione che fa predizione sul corrente allenamento
function Predizione() {
    //carico i coefficienti, medie e madiane
    coefficienti = CaricaJSON('Standardizzazione.txt');
    medie = coefficienti[0];
    std = coefficienti[1];
    coefficienti = coefficienti[2];

    //carico allenamenti
    allenamento = CalcolaAllenamento(LeggiEsercizi());

    // carico dati giocatori
    giocatori = CaricaJSON('Giocatori.txt');

    //cambio nome alle collone (tolgo '_mean')
    keys = getKeys(allenamento)
    for (i in keys) {
        key_ = keys[i];
        key = key_.split('_')[0];
        allenamento[key] = allenamento[key_];
        delete allenamento[key_];
    }
    //console.log(JSON.stringify(allenamento));

    //unisco giocatori e allenamento previsto
    db_gioc_all = []
    for (i in giocatori) {
        giocatore = giocatori[i]
        db_gioc_all.push({...giocatore,
            ...allenamento
        });
    }
    //console.log(JSON.stringify(db_gioc_all));

    //standardizzo i dati
    db_gioc_all_std = [];
    for (i in db_gioc_all) {
        record = db_gioc_all[i];

        keys = getKeys(record)
        for (i in keys) {
            key = keys[i];
            if (!(key === 'playerName')) //problema con playername
            {
                record[key] = (record[key] - medie[key]) / std[key];
            }
        }
        db_gioc_all_std.push(record);
    }
    //console.log(JSON.stringify(db_gioc_all_std));

    //faccio previsioni
    db_gioc_all_std_prev = [];
    for (i in db_gioc_all_std) {
        record = db_gioc_all_std[i];
        keys = getKeys(record)

        record['Go_score+1'] = 0

        for (i in keys) {

            key = keys[i];
            if (!(key === 'playerName')) //problema con playername
            {
                record['Go_score+1'] += record[key] * coefficienti[key];
            }
        }

        db_gioc_all_std_prev.push(record);
    }
    //console.log(JSON.stringify(db_gioc_all_std_prev));

    //standardizzazione inversa
    db_gioc_all_prev = [];
    for (i in db_gioc_all_std_prev) {
        record = db_gioc_all_std_prev[i];

        keys = getKeys(record)

        for (i in keys) {
            key = keys[i];
            if (!(key === 'playerName')) //problema con playername
            {
                record[key] = (record[key] * std[key]) + medie[key];
            }
        }
        db_gioc_all_prev.push(record);
    }
    //console.log(JSON.stringify(db_gioc_all_prev));

    return db_gioc_all_prev;
}