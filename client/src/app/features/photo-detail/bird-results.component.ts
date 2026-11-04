import { ChangeDetectionStrategy, Component, effect, inject, input, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { ApiService } from '../../core/services/api.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';
import { FixedPipe } from '../../shared/pipes/fixed.pipe';

interface BirdResult {
  status: string | null;
  score: number | null;
  count: number | null;
  best_index: number | null;
  stale: boolean;
  outdated: boolean;
  min_area_percent?: number | null;
  excluded_small_count?: number | null;
  focus_unverified_count?: number | null;
  error: string | null;
  birds: { sharpness: number | null; confidence: number; width: number; height: number;
    reason?: string | null; area_percent?: number; exclusion_reason?: string | null; focus_verified?: boolean }[];
}

@Component({
  selector: 'app-bird-results',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TranslatePipe, FixedPipe, MatButtonModule],
  template: `
    @if (result(); as r) {
      @if (r.status) {
        <section class="border-t border-[var(--mat-sys-outline-variant)] pt-3 space-y-2">
          <h3 class="font-medium">{{ 'birds.title' | translate }}</h3>
          <p>{{ ('birds.photo_status.' + r.status) | translate }}</p>
          @if (r.outdated) { <p>{{ 'birds.outdated' | translate }}</p> }
          @if (!r.outdated && r.focus_unverified_count) {
            <p class="text-sm">{{ 'birds.focus_unverified_count' | translate:{ count: r.focus_unverified_count } }}</p>
          }
          @if (!r.outdated && r.min_area_percent !== null && r.min_area_percent !== undefined) {
            <p class="text-xs opacity-70">{{ 'birds.size_rule' | translate:{ percent: r.min_area_percent } }}</p>
            @if (r.excluded_small_count) {
              <p class="text-xs opacity-70">{{ 'birds.small_excluded' | translate:{ count: r.excluded_small_count } }}</p>
            }
          }
          @if (r.stale) {
            <p>{{ 'birds.stale' | translate }}</p>
          } @else {
            @if (r.score !== null && r.score !== undefined) {
              <p>{{ 'birds.score' | translate }}: {{ r.score | fixed:2 }} / 10</p>
              <p class="text-xs opacity-70">{{ 'birds.score_hint' | translate }}</p>
            }
            @if (r.count) {
              <p>{{ 'birds.detected_count' | translate:{ count: r.count } }}</p>
              <button mat-button (click)="showPreview.set(!showPreview())">{{ 'birds.show_location' | translate }}</button>
              @if (showPreview()) {
                <img class="w-full rounded" [src]="preview()" [alt]="'birds.location_alt' | translate" />
                <p class="text-xs opacity-70">{{ 'birds.box_hint' | translate }}</p>
                <label class="block text-sm">
                  {{ 'birds.inspect_bird' | translate }}
                  <select class="ml-2 p-1 rounded bg-[var(--mat-sys-surface)]"
                          [value]="selectedBird()" (change)="selectBird($event)">
                    @for (bird of r.birds; track $index) {
                      <option [value]="$index" [selected]="$index === selectedBird()">
                        {{ 'birds.bird_number' | translate:{ number: $index + 1 } }}
                        @if (!r.outdated && bird.focus_verified === false) {
                          · {{ 'birds.focus_review' | translate }}
                        } @else if (!r.outdated && bird.sharpness !== null) { · {{ bird.sharpness | fixed:2 }} }
                      </option>
                    }
                  </select>
                </label>
                @if (selectedBird() !== null) {
                  @if (r.birds[selectedBird()!]; as bird) {
                    @if (bird.area_percent !== undefined) {
                      <p class="text-sm">{{ 'birds.size_metric' | translate:{ width: bird.width, height: bird.height, percent: (bird.area_percent | fixed:3) } }}</p>
                    }
                    @if (bird.exclusion_reason) {
                      <p class="text-sm">{{ ('birds.reason.' + bird.exclusion_reason) | translate }}</p>
                    }
                  }
                  @if (r.birds[selectedBird()!]?.reason; as reason) {
                    <p class="text-sm">{{ ('birds.reason.' + reason) | translate }}</p>
                  }
                  <img class="w-full rounded" [src]="preview(selectedBird()!)" [alt]="'birds.crop_alt' | translate" />
                }
              }
            }
          }
          @if (r.error) { <p role="alert">{{ r.error }}</p> }
        </section>
      }
    }
  `,
})
export class BirdResultsComponent {
  readonly path = input.required<string>();
  private readonly api = inject(ApiService);
  protected readonly result = signal<BirdResult | null>(null);
  protected readonly showPreview = signal(false);
  protected readonly selectedBird = signal<number | null>(null);

  constructor() {
    effect(onCleanup => {
      const path = this.path();
      this.result.set(null);
      this.showPreview.set(false);
      this.selectedBird.set(null);
      const subscription = this.api.get<BirdResult>('/birds/photo', { path }).subscribe({
        next: result => {
          this.result.set(result);
          this.selectedBird.set(result.best_index ?? (result.birds.length ? 0 : null));
        },
        error: () => this.result.set(null),
      });
      onCleanup(() => subscription.unsubscribe());
    });
  }

  protected selectBird(event: Event): void {
    const index = Number((event.target as HTMLSelectElement).value);
    if (Number.isInteger(index) && index >= 0 && index < (this.result()?.birds.length ?? 0)) {
      this.selectedBird.set(index);
    }
  }

  protected preview(index?: number): string {
    const params = new URLSearchParams({ path: this.path() });
    if (index !== undefined) params.set('index', String(index));
    return '/api/birds/preview?' + params;
  }
}
