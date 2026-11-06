import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { TestbedHarnessEnvironment } from '@angular/cdk/testing/testbed';
import { MatSelectHarness } from '@angular/material/select/testing';
import { Subject, of, throwError } from 'rxjs';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { CategoryOverrideDialogComponent } from './category-override-dialog.component';

describe('CategoryOverrideDialogComponent', () => {

  let component: any;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  let dialogClose: ReturnType<typeof vi.fn>;
  let snackOpen: ReturnType<typeof vi.fn>;

  function build(currentCategory: string | null = 'silhouette', paths?: string[], i18n: Partial<I18nService> = { t: (k: string) => k }) {
    get = vi.fn(() => of({ categories: [{ name: 'silhouette' }, { name: 'sports' }, { name: 'fashion' }] }));
    post = vi.fn(() => of({ success: true, path: '/a.jpg', old_category: 'silhouette', new_category: 'sports', aggregate: 7.42, vcg_submission_score: 7.62 }));
    dialogClose = vi.fn();
    snackOpen = vi.fn();
    TestBed.configureTestingModule({
      providers: [
        { provide: ApiService, useValue: { get, post } },
        { provide: MatSnackBar, useValue: { open: snackOpen } },
        { provide: I18nService, useValue: i18n },
        { provide: MatDialogRef, useValue: { close: dialogClose } },
        { provide: MAT_DIALOG_DATA, useValue: paths ? { paths } : { path: '/a.jpg', currentCategory } },
      ],
    });
    component = TestBed.runInInjectionContext(() => new CategoryOverrideDialogComponent());
  }

  it('loads the configured categories and preselects the current one', async () => {
    build('sports');
    await component.ngOnInit();

    expect(get).toHaveBeenCalledWith('/config/category_priorities');
    expect(component.categories()).toEqual(['silhouette', 'sports', 'fashion']);
    expect(component.selectedCategory()).toBe('sports');
    expect(component.loading()).toBe(false);
  });

  for (const batch of [false, true]) {
    it(`localizes category options and the selection in ${batch ? 'batch' : 'single-photo'} mode`, async () => {
      const translations = signal<Record<string, string>>({ 'category_names.sports': '体育' });
      build(null, batch ? ['/a.jpg', '/b.jpg'] : undefined, {
        translations, t: (key: string) => translations()[key] ?? key,
      });
      get.mockReturnValueOnce(of({ categories: [{ name: 'sports' }, { name: 'custom_category' }] }));
      const fixture = TestBed.createComponent(CategoryOverrideDialogComponent);
      fixture.detectChanges();
      await fixture.whenStable();
      const select = await TestbedHarnessEnvironment.loader(fixture).getHarness(MatSelectHarness);

      await select.open();
      const options = await select.getOptions();
      expect(await Promise.all(options.map(option => option.getText()))).toEqual(['体育', 'Custom Category']);
      await select.clickOptions({ text: '体育' });
      expect(await select.getValueText()).toBe('体育');
      expect((fixture.componentInstance as any).selectedCategory()).toBe('sports');

      translations.set({ 'category_names.sports': 'Sports' });
      fixture.detectChanges();
      expect(await select.getValueText()).toBe('Sports');
    });
  }

  it('falls back to the first category when no current category is set', async () => {
    build(null);
    await component.ngOnInit();

    expect(component.selectedCategory()).toBe('silhouette');
  });

  it('saves the override and closes with the new category and recomputed aggregate', async () => {
    build('silhouette');
    await component.ngOnInit();
    component.selectedCategory.set('sports');

    await component.save();

    expect(post).toHaveBeenCalledWith('/comparison/override_category', { path: '/a.jpg', category: 'sports' });
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', aggregate: 7.42, vcg_submission_score: 7.62 });
  });

  it('does not close on save error', async () => {
    build('silhouette');
    await component.ngOnInit();
    post.mockReturnValueOnce(throwError(() => new Error('boom')));

    await component.save();

    expect(dialogClose).not.toHaveBeenCalled();
    expect(component.saving()).toBe(false);
  });

  it('requires an explicit category choice for multiple photos', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    expect(component.selectedCategory()).toBe('');
    await component.save();
    expect(post).not.toHaveBeenCalled();
  });

  it('saves the whole selection in one request and reports the actual updated count', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    component.selectedCategory.set('sports');
    post.mockReturnValueOnce(of({ new_category: 'sports', updated: 2 }));

    await component.save();

    expect(post).toHaveBeenCalledExactlyOnceWith('/comparison/override_categories', {
      paths: ['/a.jpg', '/b.jpg'], category: 'sports',
    });
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', updated: 2 });
  });

  it('prevents closing or duplicate submissions while a batch is saving, then permits retry', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    component.selectedCategory.set('sports');
    const response = new Subject();
    post.mockReturnValueOnce(response);
    const pending = component.save();
    const ref = TestBed.inject(MatDialogRef);
    expect(ref.disableClose).toBe(true);
    await component.save();
    expect(post).toHaveBeenCalledTimes(1);
    response.error(new Error('failed'));
    await pending;
    expect(ref.disableClose).toBe(false);
    expect(component.saving()).toBe(false);
    expect(dialogClose).not.toHaveBeenCalled();

    post.mockReturnValueOnce(of({ new_category: 'sports', updated: 2 }));
    await component.save();
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', updated: 2 });
  });

  it('does not save if the configured categories could not be loaded', async () => {
    build('sports');
    get.mockReturnValueOnce(throwError(() => new Error('failed')));
    await component.ngOnInit();
    await component.save();
    expect(post).not.toHaveBeenCalled();
  });
});
