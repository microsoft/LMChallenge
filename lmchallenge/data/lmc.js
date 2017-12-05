// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// *** A Javascript visualizer for LMC logs ***


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
        "filter_included": 0,
        "inaccurate_incorrect": 0,
        "inaccurate_correct": 0,
        "accurate_incorrect": 0,
        "accurate_correct": 0
    };
    var total = 0
    results.forEach(function (d) {
        if (d.select) {
            stats.filter_included += 1;
            var before = d.verbatim === d.target;
            var after = d.results[0][0] === d.target;
            stats.inaccurate_incorrect += !before && !after;
            stats.inaccurate_correct += !before && after;
            stats.accurate_incorrect += before && !after;
            stats.accurate_correct += before && after;
        }
        ++total;
    });
    if (total !== 0) {
        stats.inaccurate_incorrect /= stats.filter_included;
        stats.inaccurate_correct /= stats.filter_included;
        stats.accurate_incorrect /= stats.filter_included;
        stats.accurate_correct /= stats.filter_included;
        stats.filter_included /= total;
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
                  .append(datum.target)
                  .append("<br/>")
                  .append($("<b>").text("Corrupted: "))
                  .append(datum.verbatim));

    var table = $("<table>")
        .addClass("table").addClass("table-hover").addClass("table-bordered").addClass("results-table")
        .append("<tr><th>Rank:</th><th>Result:</th><th>Score:</th><th>Error score:</th><th>LM score:</th></tr>");

    for (var i = 0; i < datum.results.length; ++i) {
        var d = datum.results[i];
        var rank = i + 1;
        var entry = $("<tr>").append($("<td>").text(rank))
            .append($("<td>").text(d[0]))
            .append($("<td>").text(to_fixed(d[3], 2)))  // score
            .append($("<td>").text(to_fixed(d[1], 2)))  // error score
            .append($("<td>").text(to_fixed(d[2], 2))); // lm score
        if (d[0] === datum.target) {
            entry.addClass("target-row");
        } else if (d[0] === datum.verbatim) {
            entry.addClass("verbatim-row");
        }
        table.append(entry);
    }
    detail.append(table);

    $(".detail").empty().append(detail);
}

// Return the input data, grouped by user's messages
//   data -- a list of events for tokens
//   returns -- a list of list of events for each message
function data_by_line(data) {
    var lines = [];
    var user = null;
    var message = NaN;
    var line = null;
    for (var i = 0; i < data.length; ++i) {
        var d = data[i];
        if (d.user === user && d.message === message) {
            line.push(d);
        } else {
            line = [d];
            lines.push(line);
            user = d.user;
            message = d.message;
        }
    }
    return lines;
}

function render_pretty(data) {
    var root = d3.select(".pretty");

    var rows = root.selectAll("p")
        .data(data_by_line(data))
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
            return "<div class=\"tip\"><b>Target: </b>" + d.target +
                "<br/><b>Corrupted: </b>" + d.verbatim +
                "<br/><b>Prediction: </b>" + d.results[0][0] +
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
        .text(function (d) { return d.target; });
    cells.append("div")
        .classed("word-detail", true)
        .attr("visibility", "hidden")
        .text(function (d) {
            var r0 = d.results[0][0];
            return (d.target === d.verbatim && d.verbatim === r0) ? "" : d.verbatim;
        });
    cells.append("div")
        .classed("word-detail", true)
        .attr("visibility", "hidden")
        .text(function (d) {
            var r0 = d.results[0][0];
            return (d.target === d.verbatim && d.verbatim === r0) ? "" : r0;
        });

    set_annotations($(".annotations")[0].checked);

    // Update logic - General styling
    var all_cells = root.selectAll("p").selectAll("div");
    all_cells.classed("filtered", function(d) {
        return !d.select;
    });
    all_cells.classed("uncorrected", function(d) {
        return d.select && d.target !== d.verbatim && d.target !== d.results[0][0];
    });
    all_cells.classed("miscorrected", function(d) {
        return d.select && d.target === d.verbatim && d.target !== d.results[0][0];
    });
    all_cells.classed("corrected", function(d) {
        return d.select && d.target !== d.verbatim && d.target === d.results[0][0];
    });
}

function render_summary(stats) {
    var table = $("<table>").addClass("table").addClass("table-bordered")
        .append($("<tr>")
                .append($("<td>").addClass('info').text("Filter " + percent(stats.filter_included)))
                .append("<td>Incorrect</td>")
                .append("<td>Correct</td>"))
        .append($("<tr>").append("<td>Inaccurate</td>")
                .append($("<td>").addClass("warning").text(percent(stats.inaccurate_incorrect)))
                .append($("<td>").addClass("success").text(percent(stats.inaccurate_correct))))
        .append($("<tr>").append("<td>Accurate</td>")
                .append($("<td>").addClass("danger").text(percent(stats.accurate_incorrect)))
                .append($("<td>").addClass("active").text(percent(stats.accurate_correct))));

    $(".results").empty().append(table);
}

function wr_data(results, model) {
    render_summary(summary_stats(results));
    render_pretty(results);
    $(".opt-settings").text(model);
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
