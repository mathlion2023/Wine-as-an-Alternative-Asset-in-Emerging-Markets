clear; clc;
Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
house = string(Tver.auction_house);
country = string(Tver.wine_country);

targets = ["Acker Merrall Condit","Webb`s","Winefields","Bruun Rasmussen","Bukowskis","Baghera","Steinfels","Waddingtons"];
for i=1:numel(targets)
    mask = house==targets(i);
    fprintf('\n%s : total n=%d\n', targets(i), sum(mask));
    ctab = tabulate(country(mask));
    disp(ctab);
end

fprintf('\n--- Totals by country (all houses) ---\n');
disp(tabulate(country));
