function Tver = normalize_lot_currency(Tver)
% normalize_lot_currency: adds a priceUSD column to the raw Verdickt lot
% table (Data Australia.xlsx, which pools Australia/Argentina/Chile).
%
% 2026-11-02: Dr. Gertjan Verdickt confirmed directly, on request, that
% every price in this file is denominated in Australian dollars,
% regardless of which auction house recorded the sale or which country
% the wine itself is from. This supersedes three prior rounds of
% currency treatment that had to infer the currency from each auction
% house's jurisdiction in the absence of a direct answer from the data
% provider (2026-10-08: Langton's-only vs. blanket-USD; 2026-10-22: a
% per-house EUR/GBP/ZAR map; 2026-10-29: a further six-currency FX
% extension plus a quantified, but explicitly unresolved, Acker/Sotheby's/
% Christie's/Bonhams ambiguity). With the provider's own confirmation in
% hand, every row converts via the same monthly AUDUSD rate; no per-house
% branching is needed, and the earlier house-jurisdiction inference is no
% longer the authoritative basis for this conversion.
optsFX = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsFX.VariableNamesRange = 'A6';
optsFX.DataRange          = 'A7';
Tfx = readtable('Wine DATA.xlsx', optsFX);
Tfx.Properties.VariableNames{1} = 'Date';
Tfx.Period = dateshift(datetime(Tfx.Date), 'start', 'month');
fxKeys = cellstr(datestr(Tfx.Period, 'yyyy-mm'));
fxMapAUD = containers.Map(fxKeys, num2cell(Tfx.AUDUSD));

ymStr = cellstr(datestr(Tver.ymDate, 'yyyy-mm'));
% No default of 1 here: a row whose auction month has no matching AUDUSD
% rate gets NaN, not a silent face-value AUD-as-USD assumption, so it is
% dropped by the caller's isfinite(price) filter instead of mis-priced.
fxRate = nan(height(Tver), 1);
for i = 1:height(Tver)
    if isKey(fxMapAUD, ymStr{i})
        fxRate(i) = fxMapAUD(ymStr{i});
    end
end
Tver.priceUSD = Tver.pricePerBottle .* fxRate;
end
