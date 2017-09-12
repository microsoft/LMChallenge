// *** A Javascript visualizer for some IC logs ***


// Helpers

function to_fixed(x, n) {
    return x === null ? "null" : x.toFixed(n);
}

function percent(x) {
    return to_fixed(100 * x, 1) + " %";
}

// Rendering

// Computes summary statistics from the results set, as ratios
function summary_stats(results) {
    var stats = {
        "inaccurate_incorrect": 0,
        "inaccurate_correct": 0,
        "accurate_incorrect": 0,
        "accurate_correct": 0
    };
    var total = 0
    results.forEach(function (line) {
        line.forEach(function (d) {
            var before = d.v === d.t;
            var after = d.r[0].w === d.t;
            stats.inaccurate_incorrect += !before && !after;
            stats.inaccurate_correct += !before && after;
            stats.accurate_incorrect += before && !after;
            stats.accurate_correct += before && after;
            ++total;
        });
    });
    if (total !== 0) {
        for (var k in stats) {
            stats[k] = stats[k] / total;
        }
    }
    return stats;
}

function set_annotations(visible) {
    if (visible) {
        $(".word-detail").show();
    } else {
        $(".word-detail").hide();
    }
    $(".word").tooltip(visible ? "disable" : "enable");
}

function render_detail(datum) {
    var detail = $("<div>");

    detail.append($("<p>")
                  .append($("<b>").text("Target: "))
                  .append(datum.t)
                  .append("<br/>")
                  .append($("<b>").text("Corrupted: "))
                  .append(datum.v));

    var table = $("<table>")
        .addClass("table").addClass("table-hover").addClass("table-bordered").addClass("results-table")
        .append("<tr><th>Rank:</th><th>Result:</th><th>Score:</th><th>Error score:</th><th>LM score:</th></tr>");

    for (var i = 0; i < datum.r.length; ++i) {
        var d = datum.r[i];
        var rank = i + 1;
        var entry = $("<tr>").append($("<td>").text(rank))
            .append($("<td>").text(d.w))
            .append($("<td>").text(to_fixed(d.s, 2)))
            .append($("<td>").text(to_fixed(d.e, 2)))
            .append($("<td>").text(to_fixed(d.m, 2)));
        if (d.w === datum.t) {
            entry.addClass("info");
        }
        table.append(entry);
    }
    detail.append(table);

    $(".detail").empty().append(detail);
}

function render_pretty(log) {
    var root = d3.select(".pretty");

    var rows = root.selectAll("p")
        .data(log)
        .enter()
        .append("p")
        .classed("line", true);

    var cells = rows.selectAll("div")
        .data(function (d) { return d; })
        .enter()
        .append("div")
        .classed("word", true);

    // Tooltips
    cells.attr("data-toggle", "tooltip")
        .attr("data-html", true)
        .attr("data-placement", "bottom")
        .attr("title", function (d) {
            return "<div class=\"tip\"><b>Target: </b>" + d.t +
                "<br/><b>Corrupted: </b>" + d.v +
                "<br/><b>Prediction: </b>" + d.r[0].w +
                "</div>";
        });

    // Click - details
    cells.on("click", function(d) {
        $(".word.selected").removeClass("selected");
        $(this).addClass("selected");
        render_detail(d);
        d3.event.stopPropagation();
    });

    // Content
    cells.append("div")
        .text(function (d) { return d.t; });
    cells.append("div")
        .classed("word-detail", true)
        .attr("visibility", "hidden")
        .text(function (d) {
            var r0 = d.r[0].w;
            return (d.t === d.v && d.v === r0) ? "" : d.v;
        });
    cells.append("div")
        .classed("word-detail", true)
        .attr("visibility", "hidden")
        .text(function (d) {
            var r0 = d.r[0].w;
            return (d.t === d.v && d.v === r0) ? "" : r0;
        });

    set_annotations($(".annotations")[0].checked);

    // Update logic - General styling
    var all_cells = root.selectAll("p").selectAll("div");
    all_cells.classed("uncorrected", function(d) {
        return d.t !== d.v && d.t !== d.r[0].w;
    });
    all_cells.classed("miscorrected", function(d) {
        return d.t === d.v && d.t !== d.r[0].w;
    });
    all_cells.classed("corrected", function(d) {
        return d.t !== d.v && d.t === d.r[0].w;
    });
}

function render_summary(stats) {
    var table = $("<table>").addClass("table").addClass("table-bordered")
        .append($("<tr><td></td><td>Incorrect</td><td>Correct</td></tr>"))
        .append($("<tr>").append("<td>Inaccurate</td>")
                .append($("<td>").addClass("warning").text(percent(stats.inaccurate_incorrect)))
                .append($("<td>").addClass("success").text(percent(stats.inaccurate_correct))))
        .append($("<tr>").append("<td>Accurate</td>")
                .append($("<td>").addClass("danger").text(percent(stats.accurate_incorrect)))
                .append($("<td>").addClass("active").text(percent(stats.accurate_correct))));

    $(".results").empty().append(table);
}

function render_opt_settings(opt) {
    $(".opt-settings").text(JSON.stringify(opt, null, 1));
}

function ic_set_data(results, opt) {
    render_summary(summary_stats(results));
    render_pretty(results);
    render_opt_settings(opt);
}

// A little setup on page load
$(function() {
    $(".annotations").change(function() {
        set_annotations(this.checked);
    });
    $(".pretty").click(function () {
        $(".word.selected").removeClass("selected");
        $(".detail").empty();
    });
});
